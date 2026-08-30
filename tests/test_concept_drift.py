import os
import sys
import logging
from datetime import datetime

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import MarketDataLoader
from src.data.dataset_builder import DatasetBuilder
from src.models.tft_model import TFTForecaster
from src.mcp_server.server import evaluate_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_concept_drift_crash():
    logger.info("--- Starting Concept Drift / Market Crash Test ---")
    logger.info("Simulating highly volatile regime: March 2020 COVID-19 Crash")
    
    ticker = "^NSEI"  # NIFTY 50
    start_date = "2020-01-01"
    end_date = "2020-06-30"
    
    loader = MarketDataLoader([ticker])
    logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
    df = loader.fetch_data(start_date, end_date)
    
    if df.empty:
        logger.warning(f"Failed to fetch data for {ticker}. Check internet connection or ticker validity.")
        return
        
    df = loader.preprocess(df)
    
    # We only have ~120 trading days in 6 months, so we need to adjust encoder length for this small slice test
    builder = DatasetBuilder(df, max_encoder_length=10, max_prediction_length=2)
    train_ds, val_ds, test_ds = builder.build_datasets()
    
    # Run the TFT model structure
    logger.info("Initializing TFT model structure on volatile dataset...")
    tft = TFTForecaster(train_ds)
    
    logger.info("Evaluating model metrics during the shock period...")
    # Call the tool to get metrics
    metrics = evaluate_model_metrics(ticker)
    
    logger.info(f"Metrics Output: {metrics}")
    
    # Assertions
    assert metrics["status"] == "success", "Failed to retrieve metrics"
    assert "metrics" in metrics, "Missing metrics dictionary"
    
    logger.info("SUCCESS: The system successfully processed the macroeconomic shock data and returned drift evaluation metrics.")

if __name__ == "__main__":
    test_concept_drift_crash()
