"""Evaluation module for VYORIS.

Evaluates the trained TFT model against a baseline LSTM model
on the unseen test dataset, reporting standard metrics (RMSE, MAE, MAPE, R²).
"""

import logging
import torch
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from torch import nn

from src.models.tft_model import TFTForecaster
from src.config import config

logger = logging.getLogger(__name__)


class LSTMBaseline(pl.LightningModule):
    """Simple LSTM Baseline for comparative evaluation."""
    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 2, lr: float = 0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, 1)
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x["encoder_cont"] contains continuous variables [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(x["encoder_cont"])
        # Take the last time step for prediction
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        # y is a tuple where the first element is the target tensor
        target = y[0] if isinstance(y, tuple) else y
        # Target shape: [batch_size, pred_len]. We evaluate 1-step ahead for the baseline.
        loss = self.loss_fn(preds, target[:, 0])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ModelEvaluator:
    """Evaluates and compares the TFT model against an LSTM baseline."""
    
    def __init__(self, tft_forecaster: TFTForecaster, test_dataset: TimeSeriesDataSet):
        """Initializes the evaluator.
        
        Args:
            tft_forecaster: The trained TFTForecaster instance.
            test_dataset: The test TimeSeriesDataSet.
        """
        self.tft_forecaster = tft_forecaster
        self.test_dataset = test_dataset
        self.lstm_model = None
        self.lstm_trainer = None

    def _train_lstm_baseline(self, train_ds: TimeSeriesDataSet, val_ds: TimeSeriesDataSet) -> None:
        """Trains a baseline LSTM using the same datasets.
        
        Args:
            train_ds: Training TimeSeriesDataSet.
            val_ds: Validation TimeSeriesDataSet.
        """
        logger.info("Training baseline LSTM model...")
        
        # Get input size dynamically from dataset
        example_x, _ = next(iter(train_ds.to_dataloader(train=True, batch_size=1, num_workers=0)))
        input_size = example_x["encoder_cont"].shape[-1]
        
        self.lstm_model = LSTMBaseline(input_size=input_size)
        
        train_dataloader = train_ds.to_dataloader(train=True, batch_size=config.tft_batch_size, num_workers=0)
        val_dataloader = val_ds.to_dataloader(train=False, batch_size=config.tft_batch_size * 2, num_workers=0)
        
        self.lstm_trainer = pl.Trainer(
            max_epochs=5,  # Keep baseline training short
            accelerator="auto",
            enable_model_summary=False,
            logger=False
        )
        # We don't necessarily need a validation loop for the simple baseline test
        self.lstm_trainer.fit(self.lstm_model, train_dataloaders=train_dataloader)
        logger.info("Baseline LSTM training completed.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates standard regression metrics.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of calculated metrics.
        """
        # Flatten arrays for scikit-learn metrics
        y_t = y_true.flatten()
        y_p = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        mape = mean_absolute_percentage_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        
        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "R2": float(r2)
        }

    def evaluate(self, train_ds: TimeSeriesDataSet, val_ds: TimeSeriesDataSet) -> Dict[str, Dict[str, float]]:
        """Evaluates both models on the test set and compares performance.
        
        Args:
            train_ds: Training TimeSeriesDataSet (needed to train baseline).
            val_ds: Validation TimeSeriesDataSet.
            
        Returns:
            Nested dictionary containing metrics for both 'TFT' and 'LSTM'.
        """
        if self.tft_forecaster.model is None:
            raise RuntimeError("TFT model must be trained before evaluation.")
            
        # 1. Train LSTM Baseline
        self._train_lstm_baseline(train_ds, val_ds)
        
        test_dataloader = self.test_dataset.to_dataloader(train=False, batch_size=config.tft_batch_size * 2, num_workers=0)
        
        logger.info("Evaluating TFT Model on Test Set...")
        tft_preds = self.tft_forecaster.model.predict(test_dataloader)
        
        # QuantileLoss predicts quantiles. Index 3 is typically the median for default 7 quantiles.
        if len(tft_preds.shape) == 3:
            tft_point_preds = tft_preds[:, :, 3].cpu().numpy()
        else:
            tft_point_preds = tft_preds.cpu().numpy()
            
        # Extract ground truth manually to avoid unpacking issues across versions
        y_true_list = []
        for _, y in test_dataloader:
            target = y[0] if isinstance(y, tuple) else y
            y_true_list.append(target.cpu().numpy())
        y_true = np.concatenate(y_true_list, axis=0)
        
        tft_metrics = self._compute_metrics(y_true, tft_point_preds)
        
        logger.info("Evaluating LSTM Baseline on Test Set...")
        self.lstm_model.eval()
        lstm_preds = []
        with torch.no_grad():
            for x, y in test_dataloader:
                preds = self.lstm_model(x)
                lstm_preds.append(preds.cpu().numpy())
        
        lstm_point_preds = np.concatenate(lstm_preds, axis=0)
        
        # We evaluate 1-step ahead for the baseline.
        lstm_metrics = self._compute_metrics(y_true[:, 0] if len(y_true.shape) > 1 else y_true, lstm_point_preds)
        
        comparison = {
            "TFT": tft_metrics,
            "LSTM": lstm_metrics
        }
        
        logger.info(f"Evaluation Results: {comparison}")
        return comparison
