import asyncio
import logging
import sys
import os

# Add the project root to the python path to resolve src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.agent_orchestration import agent_executor
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestAgentOrchestration")

def print_agent_trace(result: dict):
    """Parses and prints the langgraph messages trace to prove tool execution."""
    messages = result.get("messages", [])
    
    print("\n" + "="*60)
    print("AGENT EXECUTION TRACE")
    print("="*60 + "\n")
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"USER: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"AGENT CALLING TOOL: {tc['name']}")
                    print(f"   Arguments: {tc['args']}\n")
            elif msg.content:
                print(f"AGENT FINAL INSIGHT:\n\n{msg.content}\n")
        elif isinstance(msg, ToolMessage):
            print(f"TOOL RESPONSE ({msg.name}):")
            # Truncate long tool responses for readability
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "... [TRUNCATED]"
            print(f"   {content}\n")
            
    print("="*60 + "\n")

async def test_valid_ticker():
    ticker = "HINDUNILVR.NS"
    logger.info(f"--- Testing Orchestrator with VALID ticker: {ticker} ---")
    query = f"Analyze the current market sentiment and forecasting data for ticker '{ticker}'"
    
    try:
        inputs = {"messages": [("user", query)]}
        result = await agent_executor.ainvoke(inputs)
        print_agent_trace(result)
    except Exception as e:
        logger.error(f"Test failed unexpectedly: {e}")

async def test_invalid_ticker():
    ticker = "INVALIDTICKER123.XYZ"
    logger.info(f"--- Testing Orchestrator with INVALID ticker: {ticker} ---")
    query = f"Analyze the current market sentiment and forecasting data for ticker '{ticker}'"
    
    try:
        inputs = {"messages": [("user", query)]}
        result = await agent_executor.ainvoke(inputs)
        print_agent_trace(result)
    except Exception as e:
        logger.error(f"Test crashed instead of handling gracefully: {e}")

async def main():
    logger.info("Starting Agent Orchestration Integration Tests")
    await test_valid_ticker()
    await test_invalid_ticker()
    logger.info("All orchestration tests completed.")

if __name__ == "__main__":
    asyncio.run(main())
