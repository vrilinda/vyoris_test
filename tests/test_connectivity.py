import logging
import requests
import sys
import os

# Adjust path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import config
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConnectivityTest")

def test_finnhub():
    logger.info("Testing Finnhub API connectivity...")
    if not config.finnhub_api_key:
        logger.warning("Finnhub API key is not configured.")
        return False
        
    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={config.finnhub_api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Finnhub API connection successful.")
            return True
        else:
            logger.error(f"❌ Finnhub API failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Finnhub API exception: {e}")
        return False

def test_newsdata():
    logger.info("Testing NewsData.io API connectivity...")
    if not config.newsdata_api_key:
        logger.warning("NewsData API key is not configured.")
        return False
        
    url = f"https://newsdata.io/api/1/news?apikey={config.newsdata_api_key}&q=finance"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("✅ NewsData API connection successful.")
            return True
        elif response.status_code == 429:
            logger.warning("✅ NewsData API connection successful, but rate limited (429).")
            return True
        else:
            logger.error(f"❌ NewsData API failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ NewsData API exception: {e}")
        return False

def test_llm():
    logger.info("Testing LLM (Anthropic) API connectivity...")
    if not config.llm_api_key:
        logger.warning("LLM API key is not configured.")
        return False
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": config.llm_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("✅ LLM (Anthropic) API connection successful.")
            return True
        elif response.status_code == 401:
            logger.error("❌ LLM API failed: Unauthorized (invalid key).")
            return False
        elif response.status_code in [400, 404]:
             logger.warning(f"✅ LLM API connection successful but returned error (auth passed): {response.text}")
             return True
        else:
            logger.error(f"❌ LLM API failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ LLM API exception: {e}")
        return False

def test_supabase():
    logger.info("Testing Supabase connectivity (REST & DB)...")
    if not config.supabase_url or not config.supabase_key:
        logger.warning("Supabase URL or Key is not configured.")
        return False
        
    try:
        supabase: Client = create_client(config.supabase_url, config.supabase_key)
        res = supabase.table('vyoris_health_check').select("*").limit(0).execute()
        logger.info("✅ Supabase connection successful.")
        return True
    except Exception as e:
        error_str = str(e)
        if 'PGRST205' in error_str or 'Could not find the table' in error_str or '404' in error_str:
             logger.info("✅ Supabase connection successful (table not found, but DB reachable via REST).")
             return True
        logger.error(f"❌ Supabase connection exception: {e}")
        return False

def run_all_tests():
    logger.info("--- Starting VYORIS Connectivity Tests ---")
    finnhub_ok = test_finnhub()
    newsdata_ok = test_newsdata()
    llm_ok = test_llm()
    db_ok = test_supabase()
    
    logger.info("--- Connectivity Test Summary ---")
    logger.info(f"Finnhub API: {'✅' if finnhub_ok else '❌'}")
    logger.info(f"NewsData API: {'✅' if newsdata_ok else '❌'}")
    logger.info(f"LLM (Anthropic) API: {'✅' if llm_ok else '❌'}")
    logger.info(f"Supabase DB: {'✅' if db_ok else '❌'}")
    
    if all([finnhub_ok, newsdata_ok, llm_ok, db_ok]):
        logger.info("🎉 All external connections validated successfully.")
    else:
        logger.warning("⚠️ One or more external connections failed. Check configuration and network.")

if __name__ == "__main__":
    run_all_tests()
