"""End-to-end integration test for Stage 1 & 2 pipeline.

Validates configuration loading, data acquisition, preprocessing,
Dataset building, TFT initialization, training, and LSTM baseline evaluation.
"""

import logging
import os
import sys

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.data.data_loader import MarketDataLoader
from src.data.dataset_builder import DatasetBuilder
from src.models.tft_model import TFTForecaster
from src.models.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestPipeline")

def run_integration_test():
    """Executes a miniaturized end-to-end pipeline run."""
    logger.info("--- Starting Integration Test ---")
    
    # 1. Configuration Check
    logger.info(f"Loaded Environment: {config.env}")
    if not config.supabase_key:
        logger.warning("Supabase key is not loaded, but continuing test.")
    
    # Override epochs for a fast test run
    config.tft_max_epochs = 1
    config.tft_batch_size = 16
    
    # 2. Data Loading & Preprocessing
    # Using smaller constituent samples for speed ('RELIANCE.NS', 'TCS.NS')
    tickers = ['RELIANCE.NS', 'TCS.NS']
    loader = MarketDataLoader(tickers=tickers)
    
    logger.info(f"Fetching data for {tickers}...")
    # 2 years of data for a solid test set
    raw_df = loader.fetch_data(start_date="2022-01-01", end_date="2024-01-01")
    logger.info(f"Raw data shape: {raw_df.shape}")
    
    processed_df = loader.preprocess(raw_df)
    logger.info(f"Processed data shape: {processed_df.shape}")
    
    # 3. Dataset Building
    # Reduce lengths for testing purposes
    builder = DatasetBuilder(processed_df, max_encoder_length=60, max_prediction_length=5)
    train_ds, val_ds, test_ds = builder.build_datasets()
    
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Validation samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")
    
    # 4. TFT Model Initialization & Training
    tft_model = TFTForecaster(training_dataset=train_ds)
    tft_model.train(train_ds, val_ds)
    
    # 5. Baseline Evaluation
    evaluator = ModelEvaluator(tft_forecaster=tft_model, test_dataset=test_ds)
    results = evaluator.evaluate(train_ds, val_ds)
    
    logger.info("--- Integration Test Completed Successfully ---")
    logger.info(f"Final Metrics:\n{results}")

if __name__ == "__main__":
    try:
        run_integration_test()
    except Exception as e:
        logger.exception("Pipeline test failed:")
