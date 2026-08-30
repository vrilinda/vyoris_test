"""Dataset building module for VYORIS.

This module splits preprocessed data chronologically and constructs
PyTorch Forecasting TimeSeriesDataSet objects with sliding windows.
"""

import logging
import pandas as pd
from typing import Tuple, Dict, Any
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from src.config import config

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Builds Train, Validation, and Test datasets from chronological splits."""
    
    def __init__(self, df: pd.DataFrame, max_encoder_length: int = 100, max_prediction_length: int = 1):
        """Initialize the DatasetBuilder.
        
        Args:
            df: Preprocessed DataFrame containing 'ticker', 'time_idx', and numeric features.
            max_encoder_length: The lookback window size (e.g., 100 days).
            max_prediction_length: The forecast horizon.
        """
        self.df = df
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.training_dataset: TimeSeriesDataSet = None
        
    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronologically splits the data into 70% Train, 15% Validation, 15% Test.
        
        Returns:
            Tuple of DataFrames: (train_df, val_df, test_df)
        """
        logger.info("Creating chronological data splits (70/15/15)...")
        
        train_list, val_list, test_list = [], [], []
        
        # Split per ticker to maintain sequential integrity
        for ticker, group in self.df.groupby('ticker'):
            group = group.sort_values('time_idx')
            n = len(group)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])
            
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        return train_df, val_df, test_df

    def build_datasets(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """Constructs PyTorch Forecasting TimeSeriesDataSet objects.
        
        Returns:
            Tuple of TimeSeriesDataSet: (train_ds, val_ds, test_ds)
        """
        train_df, val_df, test_df = self.create_splits()
        
        logger.info(f"Building TimeSeriesDataSet with {self.max_encoder_length}-day sliding window...")
        
        # Define the training dataset schema
        self.training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="close",
            group_ids=["ticker"],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["ticker"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["open", "high", "low", "close", "volume"],
            target_normalizer=GroupNormalizer(groups=["ticker"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation and test datasets based on the training dataset's parameters
        validation_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, val_df, predict=False, stop_randomization=True)
        test_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, test_df, predict=False, stop_randomization=True)
        
        return self.training_dataset, validation_dataset, test_dataset
