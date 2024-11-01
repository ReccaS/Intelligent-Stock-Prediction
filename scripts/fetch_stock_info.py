# scripts/fetch_stock_info.py

import yfinance as yf
import pandas as pd
from typing import Optional
import logging
from datetime import datetime

def fetch_stock_data(
    ticker: str = 'SPY',
    start_date: str = '2024-01-01',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker symbol between specified dates.
    Ensures that ticker symbols are removed from column names if included.

    Parameters:
    - ticker (str, optional): The stock ticker symbol (default is 'SPY').
    - start_date (str, optional): The start date in 'YYYY-MM-DD' format (default is '2024-01-01').
    - end_date (Optional[str], optional): The end date in 'YYYY-MM-DD' format.
      If not provided, defaults to the current date.

    Returns:
    - pd.DataFrame: A DataFrame containing the stock data with simplified column names.
    """
    logger = logging.getLogger(__name__)  # Get a logger for this module

    try:
        # If end_date is not provided, set it to today's date
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            logger.info(f"No end_date provided. Using current date: {end_date}")
        else:
            logger.info(f"End date provided: {end_date}")

        logger.info(f"Fetching data for ticker: {ticker} from {start_date} to {end_date}")
        
        # Download data with ticker as a string to get single-level or multi-level columns
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data found for ticker '{ticker}' between {start_date} and {end_date}.")
            return pd.DataFrame()

        # Reset the index to make Date a column
        logger.info("Resetting index to make 'Date' a column.")
        data = data.reset_index()

        # Flatten column names if they are multi-level (i.e., contain the ticker symbol)
        if isinstance(data.columns, pd.MultiIndex):
            logger.info("Flattening multi-level column names (removing ticker symbol).")
            data.columns = data.columns.get_level_values(0)  # Keep only the first level (e.g., 'Close')

        logger.info(f"Successfully fetched and simplified data for ticker '{ticker}'.")
        return data

    except Exception as e:
        logger.error(f"Error fetching data for ticker '{ticker}': {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure
