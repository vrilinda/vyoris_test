import os
import requests
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from src.config import config

logger = logging.getLogger("SyncSymbols")

def sync_symbols_to_supabase():
    """
    Checks Supabase stock_symbols table. If empty or older than 24 hours,
    fetches all NSE equities from the official NSE CSV, formats them with .NS, 
    and bulk upserts them to Supabase.
    """
    if not config.supabase_url or not config.supabase_key:
        logger.warning("Supabase URL or Key not configured. Skipping sync.")
        return

    try:
        # Use service role key to bypass RLS for this background daemon, fallback to anon key
        key_to_use = config.supabase_service_role_key or config.supabase_key
        if not config.supabase_service_role_key:
            logger.warning("No supabase_service_role_key found. Syncing using anon key (will fail if RLS is enabled on stock_symbols!).")
        
        supabase: Client = create_client(config.supabase_url, key_to_use)
        
        # Check latest timestamp
        try:
            res = supabase.table("stock_symbols").select("last_updated").order("last_updated", desc=True).limit(1).execute()
        except Exception as e:
            logger.error(f"Error checking stock_symbols table (Ensure you have created it in Supabase!): {e}")
            return
            
        needs_sync = False
        if not res.data:
            needs_sync = True
            logger.info("Supabase table stock_symbols is empty. Triggering initial sync.")
        else:
            latest = res.data[0].get("last_updated")
            if latest:
                # Convert string to datetime
                try:
                    # Supabase returns ISO format
                    latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - latest_dt) > timedelta(hours=24):
                        needs_sync = True
                        logger.info("Supabase data is older than 24 hours. Triggering sync.")
                except Exception as e:
                    logger.error(f"Error parsing date {latest}: {e}")
                    needs_sync = True

        if not needs_sync:
            logger.info("Supabase stock_symbols is up to date. Skipping sync.")
            return

        # Fetch from NSE official CSV
        logger.info("Fetching symbols directly from NSE CSV...")
        csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        
        try:
            # We must use a User-Agent, NSE sometimes blocks raw python-requests
            storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            df = pd.read_csv(csv_url, storage_options=storage_options)
        except Exception as e:
            logger.error(f"Failed to fetch CSV from NSE: {e}")
            return
            
        # Ensure we have the required columns
        if 'SYMBOL' not in df.columns or 'NAME OF COMPANY' not in df.columns:
            logger.error(f"NSE CSV format changed. Columns found: {df.columns.tolist()}")
            return
            
        # Prepare for upsert
        now_iso = datetime.now(timezone.utc).isoformat()
        records_to_upsert = []
        
        for _, row in df.iterrows():
            raw_symbol = str(row['SYMBOL']).strip()
            company_name = str(row['NAME OF COMPANY']).strip()
            
            if raw_symbol and raw_symbol != 'nan':
                records_to_upsert.append({
                    "symbol": f"{raw_symbol}.NS",
                    "company_name": company_name,
                    "exchange": "NS",
                    "last_updated": now_iso
                })
                
        # Bulk Upsert in batches to avoid payload limits
        batch_size = 1000
        logger.info(f"Upserting {len(records_to_upsert)} records to Supabase in batches of {batch_size}...")
        
        for i in range(0, len(records_to_upsert), batch_size):
            batch = records_to_upsert[i:i + batch_size]
            supabase.table("stock_symbols").upsert(batch).execute()
            
        logger.info("Sync to Supabase completed successfully.")
        
    except Exception as e:
        logger.error(f"Failed to sync symbols to Supabase: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_symbols_to_supabase()
