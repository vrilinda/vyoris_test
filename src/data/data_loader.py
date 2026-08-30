"""Data acquisition and preprocessing module for VYORIS.

This module provides robust data loading from Yahoo Finance and
pure functions for handling missing values, outliers, and scaling.
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

from src.config import config

logger = logging.getLogger(__name__)

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills then backward-fills missing data to ensure continuity.
    
    Args:
        df: Raw pandas DataFrame.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    return df.ffill().bfill()


def handle_outliers(df: pd.DataFrame, columns: List[str], z_thresh: float = 3.0) -> pd.DataFrame:
    """Clips outliers based on Z-score thresholding.
    
    Args:
        df: Cleaned pandas DataFrame.
        columns: List of column names to process.
        z_thresh: Z-score threshold for clipping.
        
    Returns:
        DataFrame with outliers clipped.
    """
    df_out = df.copy()
    for col in columns:
        if col in df_out.columns:
            mean = df_out[col].mean()
            std = df_out[col].std()
            lower_bound = mean - z_thresh * std
            upper_bound = mean + z_thresh * std
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
    return df_out


class MarketDataLoader:
    """Handles fetching and preprocessing of historical OHLCV data."""
    
    def __init__(self, tickers: List[str]):
        """Initialize the data loader.
        
        Args:
            tickers: List of Yahoo Finance ticker symbols (e.g., '^NSEI', '^BSESN').
        """
        self.tickers = tickers
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches historical market data for all configured tickers.
        
        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            
        Returns:
            Concatenated DataFrame with an added 'ticker' column.
        """
        all_data = []
        for ticker in self.tickers:
            try:
                logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                
                # Flatten MultiIndex columns if present (yfinance behavior changes)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df['ticker'] = ticker
                all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                
        if not all_data:
            raise ValueError("No data could be fetched for any ticker.")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Applies missing value handling, outlier clipping, and scaling.
        
        Args:
            df: Raw DataFrame from fetch_data.
            fit_scaler: Whether to fit the StandardScaler or just transform.
            
        Returns:
            Preprocessed DataFrame with scaled features.
        """
        df = handle_missing_data(df)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        # Ensure columns exist
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        df = handle_outliers(df, numeric_cols)
        
        if fit_scaler:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform.")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
        # Add integer time index required by PyTorch Forecasting
        # Group by ticker and assign sequential integers
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        df['time_idx'] = df.groupby('ticker').cumcount()
        
        return df
