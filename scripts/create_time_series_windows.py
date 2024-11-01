import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_time_series_windows(data, target_column, n_timesteps):
    """
    Creates sliding windows of data for time series prediction, including the target column (e.g., Close_delta)
    in the feature set but shifting the target y to predict the next Close_delta value.

    Parameters:
    - data (pd.DataFrame): The original DataFrame containing all the features and the target.
    - target_column (str): The name of the target column (e.g., 'Close_delta').
    - n_timesteps (int): The number of timesteps (sequence length) for each window.

    Returns:
    - X (np.ndarray): The array of feature windows, including Close_delta.
    - y (np.ndarray): The array of target values (Close_delta for the next time step).
    """
    X = []
    y = []

    # Loop through the data to create windows
    for i in range(len(data) - n_timesteps):
        # Include the target column in X (do not drop Close_delta)
        X_window = data.iloc[i:i + n_timesteps].values  # Include all features, including Close_delta
        y_window = data.iloc[i + n_timesteps][target_column]  # The target value is the next Close_delta
        
        X.append(X_window)
        y.append(y_window)

    return np.array(X), np.array(y)

def prepare_data_for_training_with_windows(transformed_with_history_df: pd.DataFrame, target_column: str, n_timesteps: int):
    """
    Prepares the transformed DataFrame by generating time series windows and splitting them
    into training and testing sets with an 80/20 ratio.
    
    Parameters:
    - transformed_with_history_df (pd.DataFrame): The DataFrame containing the transformed historical data.
    - target_column (str): The name of the target column (e.g., 'Close_delta').
    - n_timesteps (int): The number of timesteps (sequence length) for each window.
    
    Returns:
    - X_train (np.ndarray): Training set features.
    - X_test (np.ndarray): Testing set features.
    - y_train (np.ndarray): Training set target.
    - y_test (np.ndarray): Testing set target.
    """
    # Step 1: Create time series windows
    X, y = create_time_series_windows(transformed_with_history_df, target_column, n_timesteps)
    
    # Step 2: Randomly split the windows into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test
