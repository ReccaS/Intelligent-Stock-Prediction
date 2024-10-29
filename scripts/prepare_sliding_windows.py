import pandas as pd

def prepare_sliding_window_data(transformed_data_df: pd.DataFrame):
    """
    Prepares the data by dropping the specified columns and keeping only the delta values.
    
    Parameters:
    - transformed_data_df (pd.DataFrame): The DataFrame containing the transformed historical data.

    Returns:
    - pd.DataFrame: The DataFrame containing only the delta values.
    """
    # Columns to drop
    columns_to_drop = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Drop the specified columns and return only delta values
    delta_columns_df = transformed_data_df.drop(columns=columns_to_drop, errors='ignore')
    
    return delta_columns_df
