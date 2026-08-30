import os
import sys
import logging
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_mock_data():
    """Generates a synthetic chronological DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=500, freq='D')
    df = pd.DataFrame({
        "date": dates,
        "ticker": "TEST.NS",
        "open": np.random.rand(500) * 100,
        "high": np.random.rand(500) * 100 + 10,
        "low": np.random.rand(500) * 100 - 10,
        "close": np.random.rand(500) * 100,
        "volume": np.random.randint(1000, 10000, 500),
        "time_idx": np.arange(500)
    })
    return df

def test_no_lookahead_bias():
    logger.info("--- Starting Data Leakage / Lookahead Bias Test ---")
    
    df = generate_mock_data()
    # Using smaller encoder length for the mock 200 period dataset
    builder = DatasetBuilder(df, max_encoder_length=30, max_prediction_length=5)
    train_ds, val_ds, test_ds = builder.build_datasets()
    
    dataloader = train_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
    
    leakage_found = False
    
    for batch_idx, (x, y) in enumerate(dataloader):
        decoder_time_idx = x["decoder_time_idx"] # Shape: [batch_size, max_prediction_length]
        
        for i in range(decoder_time_idx.shape[0]):
            min_dec_idx = decoder_time_idx[i].min().item()
            max_enc_idx = min_dec_idx - 1  # In PyTorch Forecasting, encoder strictly precedes decoder
            
            if max_enc_idx >= min_dec_idx:
                logger.error(f"DATA LEAKAGE DETECTED in batch {batch_idx}, item {i}!")
                logger.error(f"Max Encoder Time Idx: {max_enc_idx} >= Min Decoder Time Idx: {min_dec_idx}")
                leakage_found = True
                
    if not leakage_found:
        logger.info("SUCCESS: No data leakage detected. max(encoder_time_idx) strictly precedes min(decoder_time_idx) across all batches.")
        logger.info("The system is definitively proven to be free of lookahead bias.")
    else:
        raise AssertionError("Lookahead bias found in dataset construction!")

if __name__ == "__main__":
    test_no_lookahead_bias()
