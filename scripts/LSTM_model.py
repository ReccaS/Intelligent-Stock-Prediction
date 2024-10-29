import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def lstm_model(X_train, y_train, X_test, y_test, n_timesteps, n_features, num_layers, units_per_layer, 
               learning_rate=0.001, epochs=20, batch_size=32):
    """
    LSTM model for time series prediction with configurable layers, learning rate, epochs, and batch size.

    Parameters:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target.
    - n_timesteps (int): Number of timesteps in each sequence.
    - n_features (int): Number of features in each sequence.
    - num_layers (int): Number of LSTM layers.
    - units_per_layer (list of int): Number of units in each LSTM layer.
    - learning_rate (float, optional): Learning rate for the Adam optimizer.
    - epochs (int, optional): Number of epochs for training the model.
    - batch_size (int, optional): Batch size for model training.

    Returns:
    - y_pred (np.ndarray): Predicted target values for the test set.
    """
    assert len(units_per_layer) == num_layers, "Length of `units_per_layer` must match `num_layers`"

    # Reshape the data for LSTM (3D shape: [samples, timesteps, features])
    X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Build LSTM model
    model = Sequential()

    # Add LSTM layers dynamically based on `num_layers` and `units_per_layer`
    for i in range(num_layers):
        if i == num_layers - 1:  # Last layer should not return sequences
            model.add(LSTM(units_per_layer[i], activation='relu', input_shape=(n_timesteps, n_features)))
        else:  # Intermediate layers should return sequences
            model.add(LSTM(units_per_layer[i], activation='relu', return_sequences=True))

    # Add output Dense layer
    model.add(Dense(1))

    # Compile the model with the Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train the model with the provided epochs and batch size
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    return y_pred, model
