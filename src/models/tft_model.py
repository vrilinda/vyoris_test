"""Temporal Fusion Transformer (TFT) modeling module for VYORIS.

This module wraps the PyTorch Forecasting TFT implementation,
providing interfaces for model initialization, training, and 
attention weight extraction.
"""

import logging
import torch
import pandas as pd
from typing import Dict, Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from src.config import config

logger = logging.getLogger(__name__)


class TFTForecaster:
    """Wrapper class for the Temporal Fusion Transformer model."""
    
    def __init__(self, training_dataset: TimeSeriesDataSet):
        """Initializes the TFT model based on the schema of the training dataset.
        
        Args:
            training_dataset: The TimeSeriesDataSet used for training, providing
                              the necessary schema configurations.
        """
        self.training_dataset = training_dataset
        self.model: Optional[TemporalFusionTransformer] = None
        self.trainer: Optional[pl.Trainer] = None
        
        self._build_model()
        
    def _build_model(self) -> None:
        """Constructs the TFT architecture using configuration hyperparameters."""
        logger.info("Building Temporal Fusion Transformer model...")
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=config.tft_learning_rate,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # QuantileLoss default output size (7 quantiles)
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
        
    def train(self, train_ds: TimeSeriesDataSet, val_ds: TimeSeriesDataSet) -> None:
        """Trains the TFT model using PyTorch Lightning.
        
        Args:
            train_ds: Training TimeSeriesDataSet.
            val_ds: Validation TimeSeriesDataSet.
        """
        logger.info(f"Starting TFT training for max {config.tft_max_epochs} epochs...")
        
        train_dataloader = train_ds.to_dataloader(train=True, batch_size=config.tft_batch_size, num_workers=0)
        val_dataloader = val_ds.to_dataloader(train=False, batch_size=config.tft_batch_size * 2, num_workers=0)
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )
        lr_logger = LearningRateMonitor(logging_interval="step")
        
        self.trainer = pl.Trainer(
            max_epochs=config.tft_max_epochs,
            accelerator="auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback],
        )
        
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        logger.info("TFT training completed.")
        
    def extract_attention_weights(self, dataloader) -> Dict[str, Any]:
        """Extracts variable importance (self-attention weights) from the model.
        
        Args:
            dataloader: PyTorch DataLoader containing the data to interpret.
            
        Returns:
            Dictionary containing encoder and decoder variable importances.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        logger.info("Extracting self-attention weights and variable importance...")
        raw_predictions, x = self.model.predict(dataloader, mode="raw", return_x=True)
        interpretation = self.model.interpret_output(raw_predictions, reduction="sum")
        
        # Convert variable importances to human-readable format
        encoder_importance = dict(zip(
            self.model.encoder_variables, 
            interpretation["encoder_variables"].cpu().numpy().tolist()
        ))
        
        decoder_importance = dict(zip(
            self.model.decoder_variables, 
            interpretation["decoder_variables"].cpu().numpy().tolist()
        ))
        
        return {
            "encoder_importance": encoder_importance,
            "decoder_importance": decoder_importance
        }
