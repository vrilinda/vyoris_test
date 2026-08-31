# VYORIS - Quantitative Market Analysis Platform

VYORIS is a sophisticated quantitative market analysis platform designed to forecast stock prices and evaluate market sentiment. The platform integrates Deep Learning (Temporal Fusion Transformers), Natural Language Processing (FinBERT), and a dynamic agentic orchestration framework to provide deep financial insights, specifically tailored for the Indian Stock Market (NSE/BSE).

## Architecture & Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTMX, Vanilla CSS, Jinja2 Templates
- **Orchestration**: LangChain & Anthropic Claude
- **Data Engineering**: Pandas, yfinance, NewsData.io API
- **Database**: Supabase (PostgreSQL) with `pg_trgm` caching for lightning-fast stock search.
- **Machine Learning**: PyTorch Forecasting (TFT), HuggingFace (FinBERT)
- **Extensibility**: Model Context Protocol (MCP) server for local IDE integrations.

## Prerequisites
Ensure you have Python 3.10+ installed.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory and populate it with the following:
   ```env
   # LLM
   llm_api_key=your_anthropic_api_key
   
   # APIs
   newsdata_api_key=your_newsdata_io_api_key
   
   # Supabase
   database_url=postgresql://postgres:password@db.projectref.supabase.co:5432/postgres
   supabase_url=https://projectref.supabase.co
   supabase_key=your_supabase_anon_key
   ```

3. **Database Initialization:**
   Ensure the `stock_symbols` table is created in your Supabase project:
   ```sql
   CREATE TABLE stock_symbols (
       symbol TEXT PRIMARY KEY,
       company_name TEXT NOT NULL,
       exchange TEXT NOT NULL,
       last_updated TIMESTAMPTZ DEFAULT NOW()
   );
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   CREATE INDEX idx_stock_symbols_search ON stock_symbols USING gin(company_name gin_trgm_ops);
   ```

## Running the Application Locally

### 1. Web Interface (FastAPI + HTMX)
The primary user interface provides a fast, dynamic search bar and rich markdown-rendered insights without heavy frontend frameworks.

To start the web server:
```bash
uvicorn src.agents.agent_orchestration:app --reload
```
- The background daemon will automatically sync live NSE stock symbols from the official registry to Supabase on startup.
- Access the UI at: `http://localhost:8000`

### 2. Developer GUI (MCP Server)
VYORIS exposes its core intelligence tools via the Model Context Protocol (MCP), allowing AI IDEs (like Cursor or Antigravity) to directly execute the platform's quantitative models and sentiment analyzers.

To run the MCP Server:
```bash
mcp run src/mcp_server/server.py
```
This exposes the following tools to the local environment:
- `get_market_data`
- `analyze_news_sentiment`
- `generate_tft_forecast`
- `evaluate_model_metrics`
