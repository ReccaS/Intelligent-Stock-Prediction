# scripts/transform_stock_data.py

import pandas as pd
from typing import Optional, List
import logging

# Initialize a logger for this module
logger = logging.getLogger(__name__)

def transform_stock_data_to_delta(
    df: pd.DataFrame,
    columns_to_exclude: Optional[List[str]] = None,
    columns_to_calculate: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Transforms the stock data by excluding specified columns, keeping certain columns 
    without calculation, and calculating the proportion of increase or decrease (delta) 
    from the prior row value for specified columns. Increases are positive, decreases 
    are negative, and no change is zero.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing stock data.
    - columns_to_exclude (Optional[List[str]], optional): List of column names to drop from the DataFrame.
    - columns_to_calculate (Optional[List[str]], optional): List of column names to calculate deltas for.
      If None, all numeric columns except those in 'columns_to_exclude' and 'columns_to_keep' will be transformed.
    - columns_to_keep (Optional[List[str]], optional): List of column names to keep in the DataFrame but exclude from delta calculation.

    Returns:
    - pd.DataFrame: A DataFrame with the specified columns transformed to deltas, with certain columns excluded and others kept as-is.
    """
    logger.info("Starting transformation of stock data to deltas.")

    try:
        # Step 1: Drop columns to exclude
        if columns_to_exclude:
            logger.info(f"Dropping columns: {columns_to_exclude}")
            df = df.drop(columns=columns_to_exclude, errors='ignore')
        else:
            logger.info("No columns specified for exclusion.")
        
        # Step 2: Select columns for delta calculation
        if columns_to_calculate is None:
            # Automatically select all numeric columns except those in columns_to_keep
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            columns_to_calculate = [col for col in numeric_columns if col not in (columns_to_keep or [])]
            logger.info(f"No specific columns provided for delta calculation. Automatically selected: {columns_to_calculate}")
        else:
            # Validate that specified columns exist in the DataFrame
            missing_cols = [col for col in columns_to_calculate if col not in df.columns]
            if missing_cols:
                logger.error(f"The following specified columns for calculation are not in the DataFrame: {missing_cols}")
                raise ValueError(f"Missing columns for calculation: {missing_cols}")
            logger.info(f"Calculating deltas for specified columns: {columns_to_calculate}")

        # Step 3: Create a copy to avoid modifying the original DataFrame
        transformed_df = df.copy()

        # Step 4: Calculate delta (proportion change) for each specified column
        for col in columns_to_calculate:
            logger.info(f"Transforming column: {col}")
            # Compute the difference from the previous row and divide by the previous row's value
            transformed_df[f"{col}_delta"] = transformed_df[col].diff() / transformed_df[col].shift(1)

            # Round the delta values for better readability (optional)
            transformed_df[f"{col}_delta"] = transformed_df[f"{col}_delta"].round(4)

        # Step 5: Drop the first row as it will contain NaN values after diff()
        logger.info("Dropping the first row with NaN values after delta calculation.")
        transformed_df = transformed_df.dropna()

        logger.info("Successfully transformed stock data to deltas.")
        return transformed_df

    except Exception as e:
        logger.error(f"An error occurred while transforming stock data: {e}")
        raise  # Re-raise the exception after logging

def transform_with_history(transformed_df: pd.DataFrame, history_length: int = 25) -> pd.DataFrame:
    """
    Adds history columns to the transformed data by keeping only specific columns and
    appending the delta values for the past 'history_length' rows as new columns.

    Parameters:
    - transformed_df (pd.DataFrame): The transformed DataFrame containing delta columns.
    - history_length (int): The number of historical rows to append as new columns. Default is 25.

    Returns:
    - pd.DataFrame: A DataFrame with the historical delta values appended as new columns.
    """
    logger.info(f"Starting transformation with {history_length} rows of history.")

    # Step 1: Keep only the 'Date', 'Open_delta', 'High_delta', 'Low_delta', and 'Close_delta' columns
    columns_to_keep = ['Date', 'Open_delta', 'High_delta', 'Low_delta', 'Close_delta']
    
    # Check if the required columns are present in the DataFrame
    missing_cols = [col for col in columns_to_keep if col not in transformed_df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols} in transformed_df")
        raise ValueError(f"Required columns are missing: {missing_cols}")

    # Create a new DataFrame with the selected columns
    transformed_with_history_df = transformed_df[columns_to_keep].copy()

    # Step 2: Append historical columns using shift
    for i in range(1, history_length):
        logger.info(f"Adding historical columns for lag: {i}")
        
        # Shift each delta column by i rows and create new column names
        transformed_with_history_df[f'Open_delta-{i}'] = transformed_df['Open_delta'].shift(i)
        transformed_with_history_df[f'High_delta-{i}'] = transformed_df['High_delta'].shift(i)
        transformed_with_history_df[f'Low_delta-{i}'] = transformed_df['Low_delta'].shift(i)
        transformed_with_history_df[f'Close_delta-{i}'] = transformed_df['Close_delta'].shift(i)

    # Step 3: Drop rows that have NaN values due to shifting (i.e., the first 'history_length' rows)
    transformed_with_history_df = transformed_with_history_df.dropna().reset_index(drop=True)

    logger.info("Successfully transformed data with historical columns.")
    
    return transformed_with_history_df