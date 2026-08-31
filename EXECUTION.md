# Execution & Architecture Structure

This document provides a detailed breakdown of the internal mechanics of the VYORIS platform, explaining the logic of each module, code file, and their respective classes and functions.

## 1. Core Configuration (`src/config.py`)
Centralized configuration management utilizing `pydantic_settings`.
- **`Settings` Class**: Defines strictly-typed environment variables mapped from the `.env` file (e.g., Supabase credentials, API keys, TFT hyperparameters). 
- **`config` Instance**: A global singleton instance of `Settings` imported across the application to ensure uniform credential and variable access.

## 2. Orchestration & Web Server (`src/agents/agent_orchestration.py`)
The primary entry point for the FastAPI web application and LangChain agent executor.
- **FastAPI Lifecycle Hooks**: Uses `@app.on_event("startup")` to spin up a daemon thread that calls `sync_symbols_to_supabase()` in the background to ensure data freshness without blocking server startup.
- **`/api/v1/search_stocks`**: An endpoint optimized for HTMX that queries the Supabase database using ILIKE for fast, as-you-type ticker auto-completion.
- **`/api/v1/analyze`**: The core AI orchestration endpoint. It receives a ticker symbol, constructs a prompt requesting 100 days of OHLCV data and sentiment analysis, and invokes the LangChain Agent.
- **LangChain Integration**: Connects to the Anthropic LLM and binds the MCP tools dynamically using `bind_tools`.

## 3. Data Ingestion & Engineering
Data pipelines for ingesting stock lists, historical price data, and news.

### `src/data/sync_symbols.py`
Bypasses API paywalls by fetching official registries directly.
- **`sync_symbols_to_supabase()`**: Uses `pandas` to download the `EQUITY_L.csv` file directly from the NSE Archives. It extracts symbols, appends the `.NS` suffix required by `yfinance`, and bulk upserts the dataset into the Supabase `stock_symbols` table in batches of 1,000. It checks the `last_updated` timestamp to ensure syncing only occurs if data is older than 24 hours.

### `src/data/data_loader.py`
Handles raw market data fetching and preprocessing.
- **`MarketDataLoader` Class**:
  - `fetch_data()`: Uses `yfinance` to download historical OHLCV data for given tickers over a specific date range.
  - `preprocess()`: Cleans the data by forward/backward filling missing values, clipping outliers using Z-score thresholding, and applying `StandardScaler` to normalize numeric features.

### `src/data/dataset_builder.py`
Prepares DataFrames for PyTorch models.
- **`DatasetBuilder` Class**: Wraps the `TimeSeriesDataSet` from `pytorch-forecasting`, defining time index columns, targets, static reals, and time-varying features necessary for the Temporal Fusion Transformer.

## 4. MCP Server & Tools (`src/mcp_server/server.py`)
Exposes VYORIS capabilities as independent tools to any MCP-compatible client.
- **`get_market_data`**: Invokes `MarketDataLoader` to pull and summarize recent price data.
- **`analyze_news_sentiment`**: Queries the `NewsData.io` API for exact title matches (`qInTitle`) of a ticker, then passes the headlines into a local HuggingFace `FinBERT` pipeline to calculate a weighted numerical sentiment score (-1 to +1).
- **`generate_tft_forecast`**: Outlines the forecast generation pipeline using historical context to project forward horizons and extract attention weights.
- **`evaluate_model_metrics`**: Summarizes simulated model performance metrics (RMSE, MAE, MAPE, R2).

## 5. Machine Learning Models
Deep learning architectures for quantitative forecasting.

### `src/models/tft_model.py`
- **`TFTForecaster` Class**: Initializes the `TemporalFusionTransformer` using `pytorch-forecasting`. Manages the PyTorch Lightning `Trainer` for fitting the model over the dataset and defines learning rates and epoch constraints.

### `src/models/evaluator.py`
- **`ModelEvaluator` Class**: Responsible for calculating mathematical evaluation metrics (MAPE, RMSE, R^2) comparing model predictions against actual market ground truth.

## 6. Frontend Presentation
The UI is strictly separated into modular HTML and CSS to maintain a formal, academic aesthetic without heavy JavaScript frameworks.
- **`src/templates/base.html`**: The master Jinja2 layout containing the HTMX script imports, meta tags, and global layout structure.
- **`src/templates/index.html`**: Extends `base.html`. Contains the dynamic search bar that triggers HTMX requests against the `/api/v1/search_stocks` endpoint.
- **`src/templates/insight_partial.html`**: The HTML fragment returned by the `/api/v1/analyze` endpoint. It uses the `markdown2` library to render the LLM's response into clean, styled prose.
- **`src/static/styles.css`**: Contains dynamic CSS variables for theme switching (e.g., Forest Green accents) and sets the formal typography required for the academic presentation.
