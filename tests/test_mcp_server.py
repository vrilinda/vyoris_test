import json
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the python path to resolve src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the tool functions directly from the MCP server implementation
from src.mcp_server.server import (
    get_market_data,
    analyze_news_sentiment,
    generate_tft_forecast,
    evaluate_model_metrics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestMCPServer")

def run_tool_tests():
    ticker = "HINDUNILVR.NS"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"--- Testing MarketDataTool for {ticker} ({start_str} to {end_str}) ---")
    market_data_result = get_market_data(ticker=ticker, start_date=start_str, end_date=end_str)
    print(json.dumps(market_data_result, indent=2))
    print("\n" + "="*50 + "\n")

    logger.info(f"--- Testing NewsSentimentTool for {ticker} ---")
    news_sentiment_result = analyze_news_sentiment(ticker=ticker)
    print(json.dumps(news_sentiment_result, indent=2))
    print("\n" + "="*50 + "\n")

    logger.info(f"--- Testing TFTForecastTool for {ticker} ---")
    tft_forecast_result = generate_tft_forecast(ticker=ticker, forecast_horizon=5)
    print(json.dumps(tft_forecast_result, indent=2))
    print("\n" + "="*50 + "\n")

    logger.info(f"--- Testing MetricsEvaluationTool for {ticker} ---")
    metrics_eval_result = evaluate_model_metrics(ticker=ticker)
    print(json.dumps(metrics_eval_result, indent=2))
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    logger.info("Starting MCP Server Tool Tests (Direct Invocation)")
    run_tool_tests()
    logger.info("All tool tests completed.")
