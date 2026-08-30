import asyncio
import logging
import time
from httpx import AsyncClient, ASGITransport
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agent_orchestration import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def fetch_insight(client, ticker):
    start_time = time.time()
    try:
        response = await client.post(
            "/api/v1/analyze",
            json={"ticker": ticker},
            timeout=120.0
        )
        latency = time.time() - start_time
        if response.status_code == 200:
            logger.info(f"[{ticker}] SUCCESS: Latency {latency:.2f} seconds")
            return latency
        else:
            logger.error(f"[{ticker}] ERROR: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"[{ticker}] EXCEPTION: {str(e)} after {latency:.2f} seconds")
        return None

async def run_latency_test():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    logger.info(f"--- Starting E2E Latency Test with {len(tickers)} concurrent requests ---")
    
    total_start_time = time.time()
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        tasks = [fetch_insight(client, t) for t in tickers]
        results = await asyncio.gather(*tasks)
        
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    successful_latencies = [r for r in results if r is not None]
    
    logger.info("--- Latency Test Results ---")
    logger.info(f"Total Test Duration: {total_duration:.2f} seconds")
    logger.info(f"Successful Requests: {len(successful_latencies)} / {len(tickers)}")
    
    if successful_latencies:
        avg_latency = sum(successful_latencies) / len(successful_latencies)
        max_latency = max(successful_latencies)
        min_latency = min(successful_latencies)
        logger.info(f"Average Request Latency: {avg_latency:.2f} seconds")
        logger.info(f"Max Request Latency: {max_latency:.2f} seconds")
        logger.info(f"Min Request Latency: {min_latency:.2f} seconds")
        
    # Ensure at least some requests succeeded to consider the test passed
    assert len(successful_latencies) > 0, "All requests failed during the latency test."

if __name__ == "__main__":
    asyncio.run(run_latency_test())
