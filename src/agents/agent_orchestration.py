import os
import json
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Form, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

# VYORIS Imports
from src.config import config
from src.mcp_server.server import (
    get_market_data,
    analyze_news_sentiment,
    generate_tft_forecast,
    evaluate_model_metrics
)
from src.data.sync_symbols import sync_symbols_to_supabase
from supabase import create_client, Client

from src.agents.auth_router import router as auth_router, get_current_user, require_auth, get_supabase
from src.agents.history_router import router as history_router, enforce_history_retention

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AgentOrchestrator")

# Initialize FastAPI
app = FastAPI(
    title="VYORIS Agent Orchestrator",
    description="FastAPI backend for VYORIS LangChain Agent that synthesizes TFT and FinBERT data.",
    version="1.0.0"
)

# Initialize Jinja2 Templates
templates = Jinja2Templates(directory="src/templates")

# Mount Static Files
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Include Routers
app.include_router(auth_router)
app.include_router(history_router)

try:
    from markdown_it import MarkdownIt
    # explicitly enable tables on the default parser to avoid missing linkify-it-py dependency
    md_parser = MarkdownIt().enable('table')
except ImportError:
    logger.warning("markdown-it-py not found. HTML output will not be parsed from markdown.")
    md_parser = None

# Pydantic Models for API
class OrchestrationRequest(BaseModel):
    ticker: str = Field(..., description="The ticker symbol to analyze, e.g., 'RELIANCE.NS'")

class OrchestrationResponse(BaseModel):
    ticker: str
    insight: str
    status: str

# ---------------------------------------------------------------------------
# Define LangChain Tools (Wrapping MCP Server Logic)
# ---------------------------------------------------------------------------

@tool
def tool_get_market_data(ticker: str, start_date: str, end_date: str) -> str:
    """Fetches historical OHLCV data and technical indicators for a given ticker."""
    logger.info(f"[MCP TOOL EXECUTED] get_market_data | Calling Yahoo Finance for {ticker} from {start_date} to {end_date}")
    try:
        res = get_market_data(ticker, start_date, end_date)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def tool_analyze_news_sentiment(ticker: str) -> str:
    """Fetches recent news headlines and returns FinBERT sentiment (-1.0 to 1.0)."""
    logger.info(f"[MCP TOOL EXECUTED] analyze_news_sentiment | Calling Finnhub & FinBERT for {ticker}")
    try:
        res = analyze_news_sentiment(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def tool_generate_tft_forecast(ticker: str, forecast_horizon: int = 5) -> str:
    """Executes Temporal Fusion Transformer (TFT) forward inference for price projections."""
    logger.info(f"[MCP TOOL EXECUTED] generate_tft_forecast | Running PyTorch TFT inference for {ticker} (horizon: {forecast_horizon} days)")
    try:
        res = generate_tft_forecast(ticker, forecast_horizon)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def tool_evaluate_model_metrics(ticker: str) -> str:
    """Evaluates TFT vs LSTM baseline metrics (RMSE, MAE) to monitor concept drift."""
    logger.info(f"[MCP TOOL EXECUTED] evaluate_model_metrics | Calculating Drift (RMSE/MAE) for {ticker}")
    try:
        res = evaluate_model_metrics(ticker)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

tools = [
    tool_get_market_data,
    tool_analyze_news_sentiment,
    tool_generate_tft_forecast,
    tool_evaluate_model_metrics
]

# ---------------------------------------------------------------------------
# Define Agent and Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Principal Financial AI Advisor for VYORIS. 
You are operating under the COSTAR-RISE Hybrid Prompting Framework, merging strict sequential execution (Steps) with absolute output formatting control (Response Format).

[CONTEXT]
You have access to 4 distinct tools. The quantitative model (`tool_generate_tft_forecast`) outputs "normalized" indices (e.g., -1.66). Real-world currency prices (e.g., ₹2,500 INR or $150 USD) are only available via `tool_get_market_data`. 

[OBJECTIVE]
Synthesize the raw quantitative forecasting and qualitative sentiment data into a highly professional, two-part investment briefing.

[STEPS (Chain-of-Thought Execution)]
1. GATHER: You MUST explicitly call all 4 tools: `tool_get_market_data`, `tool_analyze_news_sentiment`, `tool_generate_tft_forecast`, and `tool_evaluate_model_metrics`. Do not skip any.
2. EXTRACT & CALCULATE: Extract the actual current real-world stock price from the JSON returned by `tool_get_market_data`. Apply the normalized directional trend from the TFT forecast to calculate an estimated real-world projected price target. 
3. SYNTHESIZE: Combine your calculated price projection with the FinBERT sentiment analysis to form a cohesive, logical narrative.

[AUDIENCE & TONE]
You are writing for two entirely different audiences simultaneously. 
- Section 1 Audience: Business/Retail Investors. (Tone: Accessible, expert, actionable. ZERO machine learning jargon).
- Section 2 Audience: Quants/Developers. (Tone: Highly technical, precise, data-driven ML Engineer).

[RESPONSE FORMAT]
To conserve API tokens, your response MUST be under 400 words. 
Use clean headers, bullet points, and the required markdown table. 
You MUST format your output exactly as follows:

### SECTION 1: EXECUTIVE RETAIL BRIEFING
- Price Prediction Table: You MUST display a Markdown table predicting the next two days. The columns MUST be exactly: 
  | Current Price | [Insert Tomorrow's Date] Price | [Insert Day After Tomorrow's Date] Price | Prediction Score (1-10) |
- Table Rules: Fill in actual calculated real-world prices. The Prediction Score should be your confidence level (1-10) based on FinBERT sentiment and TFT weights.
- Market Sentiment: Explain the 'Why' using pure business language (momentum, news catalysts). No mentions of "TFT", "LSTM", or "Attention Weights".

### SECTION 2: QUANTITATIVE & ML DIAGNOSTICS
- Provide a concise bulleted list explicitly citing:
  1. TFT vs LSTM baseline performance metrics (e.g., RMSE, MAE).
  2. The specific TFT attention-weight contributions (which variables technically drove the forecast).
  3. FinBERT raw sentiment scores (-1.0 to +1.0) and Concept Drift detection status.
"""

# Initialize the LLM explicitly passing the key from config
if not config.llm_api_key:
    logger.warning("No llm_api_key found in config! The agent will fail if invoked.")

llm = ChatAnthropic(
    model="claude-sonnet-5",
    api_key=config.llm_api_key or "DUMMY_KEY_FOR_TESTS"
)

agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

# ---------------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Trigger background sync of symbols to Supabase on startup."""
    logger.info("Starting background sync for stock symbols...")
    thread = threading.Thread(target=sync_symbols_to_supabase, daemon=True)
    thread.start()

@app.get("/api/v1/search_stocks", response_class=HTMLResponse)
async def search_stocks(request: Request, ticker: str = Query("", description="Search query for company name or symbol"), user = Depends(require_auth)):
    """
    HTMX endpoint that queries Supabase for matching stock symbols and returns an HTML list.
    """
    if not ticker or len(ticker) < 2:
        return HTMLResponse(content="")
        
    try:
        token = request.cookies.get("vyoris_access_token")
        supabase = get_supabase(token)
        # ILIKE query for company name or symbol
        res = supabase.table("stock_symbols")\
            .select("symbol, company_name")\
            .or_(f"company_name.ilike.%{ticker}%,symbol.ilike.%{ticker}%")\
            .limit(10)\
            .execute()
            
        data = res.data
        if not data:
            return HTMLResponse(content="<div class='p-2 text-gray-500'>No results found</div>")
            
        # Build HTML list
        html_content = "<ul class='bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-auto'>"
        for item in data:
            symbol = item.get("symbol", "")
            company = item.get("company_name", "")
            # Return li items that could be clicked or selected
            html_content += f"""
            <li class='p-2 hover:bg-forest-50 cursor-pointer border-b border-gray-100 last:border-0' 
                onclick="document.getElementById('ticker').value='{symbol}'; document.getElementById('search-results').innerHTML='';">
                <div class='font-semibold text-forest-700'>{symbol}</div>
                <div class='text-sm text-gray-500 truncate'>{company}</div>
            </li>
            """
        html_content += "</ul>"
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error querying Supabase for search: {e}")
        return HTMLResponse(content="<div class='p-2 text-red-500'>Error searching database</div>")


@app.post("/api/v1/analyze", response_model=OrchestrationResponse)
async def analyze_ticker(request: OrchestrationRequest):
    """
    Endpoint to trigger the LangChain Agent for a specific ticker.
    """
    logger.info(f"Received orchestration request for ticker: {request.ticker}")
    
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        hundred_days_ago = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        prompt_text = f"Today's date is {today_str}. Please provide a comprehensive market forecast and sentiment analysis insight for ticker: {request.ticker}. For market data, use a start date of {hundred_days_ago} and an end date of {today_str}."
        inputs = {"messages": [("user", prompt_text)]}
        result = await agent_executor.ainvoke(inputs)
        
        # In langgraph, the final output is the last message in the 'messages' list
        final_insight = result["messages"][-1].content if result.get("messages") else "No insight generated."
        
        if isinstance(final_insight, list):
            clean_parts = []
            for item in final_insight:
                if isinstance(item, dict) and item.get("type") == "text":
                    clean_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    clean_parts.append(item)
            final_insight = "\n".join(clean_parts)

        
        return OrchestrationResponse(
            ticker=request.ticker,
            insight=final_insight,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Serves the main HTMX Dashboard.
    """
    user = await get_current_user(request)
    return templates.TemplateResponse(request=request, name="index.html", context={"user": user})

@app.post("/htmx/analyze", response_class=HTMLResponse)
async def htmx_analyze_ticker(request: Request, ticker: str = Form(...), user = Depends(require_auth)):
    """
    HTMX endpoint that triggers the agent and returns a formatted HTML fragment.
    """
    logger.info(f"Received HTMX orchestration request for ticker: {ticker}")
    
    try:
        today_str = datetime.now().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        prompt_text = f"Today's date is {today_str}. Please provide a comprehensive market forecast and sentiment analysis insight for ticker: {ticker}. For market data, use a start date of {thirty_days_ago} and an end date of {today_str}."
        inputs = {"messages": [("user", prompt_text)]}
        result = await agent_executor.ainvoke(inputs)
        
        final_insight = result["messages"][-1].content if result.get("messages") else "No insight generated."
        
        if isinstance(final_insight, list):
            clean_parts = []
            for item in final_insight:
                if isinstance(item, dict) and item.get("type") == "text":
                    clean_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    clean_parts.append(item)
            final_insight = "\n".join(clean_parts)
            
        # Parse markdown to HTML
        if md_parser:
            final_insight_html = md_parser.render(final_insight)
        else:
            final_insight_html = f"<pre>{final_insight}</pre>"
            
        # Save to history
        try:
            token = request.cookies.get("vyoris_access_token")
            supabase = get_supabase(token)
            supabase.table("search_history").insert({
                "user_id": user.id,
                "ticker": ticker,
                "insight_result": final_insight_html,
                "is_favorite": False
            }).execute()
            # Enforce retention policy
            enforce_history_retention(user.id, supabase)
        except Exception as db_err:
            logger.error(f"Failed to save search history: {db_err}")
            
        return templates.TemplateResponse(
            request=request,
            name="insight_partial.html",
            context={
                "ticker": ticker,
                "insight": final_insight_html
            }
        )
    except Exception as e:
        logger.error(f"Error during HTMX agent execution: {e}")
        error_html = f"<div class='p-4 bg-red-900/50 text-red-400 rounded-lg border border-red-800'>Error: {str(e)}</div>"
        return HTMLResponse(content=error_html, status_code=500)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting VYORIS Agent Orchestrator on {config.mcp_server_host}:{config.mcp_server_port}")
    uvicorn.run("src.agents.agent_orchestration:app", host=config.mcp_server_host, port=config.mcp_server_port, reload=True)
