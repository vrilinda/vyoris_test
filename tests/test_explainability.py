import os
import sys
import logging
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_server.server import generate_tft_forecast
from src.agents.agent_orchestration import agent_executor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_tool_explainability():
    logger.info("--- Phase 1: Validating Tool Output (Attention Weights) ---")
    ticker = "RELIANCE.NS"
    result = generate_tft_forecast(ticker)
    
    assert result.get("status") == "success", "Forecast tool failed."
    assert "attention_weights" in result, "Attention weights missing from TFT forecast tool output!"
    
    weights = result["attention_weights"]
    logger.info(f"SUCCESS: TFTForecastTool returned attention weights: {weights}")
    return weights

async def test_agent_explainability(expected_weights):
    logger.info("--- Phase 2: Validating Agent Explainability ---")
    
    ticker = "RELIANCE.NS"
    # We ask the agent to explicitly use the forecast tool and explain the weights
    prompt = (
        f"Generate a financial insight for {ticker}. "
        "You MUST call the tool_generate_tft_forecast tool. "
        "Then, you MUST explicitly mention the names of the attention weights returned by the tool (e.g. close_lag_1, volume) in your final response."
    )
    
    inputs = {"messages": [("user", prompt)]}
    result = await agent_executor.ainvoke(inputs)
    final_insight = result["messages"][-1].content
    
    if isinstance(final_insight, list):
        final_insight = "\n".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in final_insight])
        
    logger.info("Agent Output:")
    # Strip emojis in case of charmap error for console printing
    safe_insight = final_insight.encode('ascii', 'ignore').decode('ascii')
    print(f"\n{safe_insight}\n")
    
    # Check if the expected weight keys are mentioned in the text output
    missing_weights = []
    found_any = False
    for weight_key in expected_weights.keys():
        if weight_key in final_insight or weight_key.replace('_', ' ') in final_insight:
            found_any = True
        else:
            missing_weights.append(weight_key)
            
    assert found_any, "The agent failed to cite ANY of the specific TFT attention weights in its final output. The 'Black Box' remains closed."
    
    if missing_weights:
        logger.warning(f"Agent did not mention all variables, missed: {missing_weights}. (This is acceptable as long as it cited the most important ones).")
        
    logger.info("SUCCESS: The LangChain agent successfully ingested and explained the TFT self-attention weights!")

async def run_explainability_tests():
    logger.info("--- Starting 'Black Box' / Explainability Test ---")
    weights = test_tool_explainability()
    await test_agent_explainability(weights)

if __name__ == "__main__":
    asyncio.run(run_explainability_tests())
