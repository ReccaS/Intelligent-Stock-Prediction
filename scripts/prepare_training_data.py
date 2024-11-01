from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data_for_training(transformed_with_history_df: pd.DataFrame):
    """
    Prepares the transformed DataFrame by dropping the original High and Low columns (not delta ones)
    and splitting the data into training and testing sets with an 80/20 ratio.
    
    Returns both feature sets (X) and target sets (y).
    
    Parameters:
    - transformed_with_history_df (pd.DataFrame): The DataFrame containing the transformed historical data.

    Returns:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.Series): Training set target.
    - y_test (pd.Series): Testing set target.
    """
    # Step 1: Drop only the original 'High' and 'Low' columns (not the delta columns)
    columns_to_drop = ['High_delta', 'Low_delta']  # Exact column names to drop
    cleaned_df = transformed_with_history_df.drop(columns=columns_to_drop, errors='ignore')

    # Step 2: Extract Close_delta as the target variable
    X = cleaned_df.drop(columns=['Close_delta'])  # Features (all columns except 'Close_delta')
    y = cleaned_df['Close_delta']  # Target variable

    # Step 3: Split the cleaned DataFrame into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
