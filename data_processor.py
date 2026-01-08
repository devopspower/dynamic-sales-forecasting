import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
    A custom PyTorch Dataset for sequential data.
    """
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_sequences(data, window_size):
    """
    Converts a flat time-series array into a supervised learning format.
    X: sequences of 'window_size' length
    y: the value immediately following the sequence
    """
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:(i + window_size)]
        y = data[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_processed_data(file_path, store_id=1, item_id=1, window_size=30, batch_size=16):
    """
    Orchestrates the data loading, filtering, scaling, and sequence creation.
    """
    # Load and Filter
    df = pd.read_csv(file_path)
    df = df[(df['store'] == store_id) & (df['item'] == item_id)]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Extract sales values and scale
    # LSTMs are sensitive to scale; MinMaxScaler is standard for RNNs
    series = df['sales'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)
    
    # Create windows
    X, y = create_sequences(scaled_data, window_size)
    
    # Sequential Split: 80% Train, 20% Validation
    # NOTE: Shuffling is FALSE to preserve chronological order
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create DataLoaders
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler