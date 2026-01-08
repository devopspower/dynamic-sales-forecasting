import torch
import torch.nn as nn

class SalesLSTM(nn.Module):
    """
    LSTM-based Recurrent Neural Network for Time-Series Forecasting.
    Designed to capture temporal dependencies in daily sales data.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(SalesLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer: batch_first=True means input shape is (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer: Maps the hidden state of the last time step to a single prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass logic.
        Args:
            x (tensor): Input tensor of shape (batch_size, window_size, input_dim)
        Returns:
            out (tensor): Predicted sales value for the next time step.
        """
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate through LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step only
        out = self.fc(out[:, -1, :])
        
        return out