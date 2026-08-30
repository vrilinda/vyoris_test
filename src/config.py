"""Configuration management for VYORIS.

This module uses Pydantic to validate and load environmental variables,
providing a strongly-typed configuration object for the entire application.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

import os
from pathlib import Path

# Get the absolute path to the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class Settings(BaseSettings):
    """Global configuration settings for VYORIS.

    Attributes:
        env: The environment (e.g., "dev", "prod", "test").
        database_url: Connection string for the PostgreSQL (Supabase) database.
        supabase_url: Supabase project URL.
        supabase_key: Supabase API Key.
        finnhub_api_key: API key for fetching financial news from Finnhub.
        newsdata_api_key: API key for fetching financial news from NewsData.io.
        openai_api_key: API key for LangChain orchestrator.
        tft_learning_rate: Learning rate for the Temporal Fusion Transformer.
        tft_batch_size: Batch size for training the TFT.
        tft_max_epochs: Maximum epochs for training.
        mcp_server_host: Host for the MCP server.
        mcp_server_port: Port for the MCP server.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # Application Settings
    env: str = Field(default="dev", description="Application environment")
    
    # API Keys & Connections (aligned with Proposal Section 8)
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/vyoris", 
        description="PostgreSQL connection string (Supabase)"
    )
    supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
    supabase_key: Optional[str] = Field(default=None, description="Supabase API Key")
    
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API Key for sentiment")
    newsdata_api_key: Optional[str] = Field(default=None, description="NewsData.io API Key for sentiment")
    newsdata_webhook_url: Optional[str] = Field(default=None, description="NewsData.io Webhook URL")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API Key for orchestration")
    
    # Model Hyperparameters
    tft_learning_rate: float = Field(default=0.03, description="TFT learning rate")
    tft_batch_size: int = Field(default=64, description="TFT batch size")
    tft_max_epochs: int = Field(default=100, description="TFT max epochs")
    
    # Server Configuration
    mcp_server_host: str = Field(default="0.0.0.0", description="MCP Server Host")
    mcp_server_port: int = Field(default=8000, description="MCP Server Port")


# Singleton instance to be imported across the application
config = Settings()
