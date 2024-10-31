import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)
        
        # Get the last time step output
        out = self.fc(out[:, -1, :])
        return out

def train_gru_model(X_train, y_train, X_test, y_test, n_timesteps, n_features, 
                    num_layers=2, units_per_layer=64, learning_rate=0.001, 
                    epochs=20, batch_size=32):
    """
    PyTorch GRU model for time series prediction.

    Parameters:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target.
    - n_timesteps (int): Number of timesteps in each sequence.
    - n_features (int): Number of features in each sequence.
    - num_layers (int, optional): Number of GRU layers.
    - units_per_layer (int, optional): Number of units in each GRU layer.
    - learning_rate (float, optional): Learning rate for the optimizer.
    - epochs (int, optional): Number of epochs for training.
    - batch_size (int, optional): Batch size for training.

    Returns:
    - y_pred (np.ndarray): Predicted target values for the test set.
    - train_loss_history (list): Training loss over epochs.
    - val_loss_history (list): Validation loss over epochs.
    - model: The trained PyTorch GRU model.
    """
    # Convert data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize the model
    model = GRUModel(input_size=n_features, hidden_size=units_per_layer, num_layers=num_layers, output_size=1)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to track losses
    train_loss_history = []
    val_loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Save training loss
        train_loss_history.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_loss_history.append(val_loss.item())

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    # Testing phase
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    return y_pred.numpy(), train_loss_history, val_loss_history, model
