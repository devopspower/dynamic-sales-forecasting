import torch
import torch.nn as nn
from data_processor import get_processed_data
from model import SalesLSTM
from train import train_model

# --- Configuration & Hyperparameters ---
CONFIG = {
    'file_path': 'data/train.csv',  # Kaggle Store Item Demand dataset
    'store_id': 1,             # Target specific store
    'item_id': 1,              # Target specific item
    'window_size': 30,         # Look back 30 days
    'batch_size': 16,
    'hidden_dim': 64,
    'num_layers': 2,
    'learning_rate': 0.001,
    'epochs': 30,
    'model_save_path': 'sales_lstm.pth'
}

def main():
    # 1. Define Device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting Sales Forecasting Pipeline ---")

    # 2. Ground Objectively: Data Preparation
    print("Loading and preprocessing sequence data...")
    train_loader, val_loader, scaler = get_processed_data(
        file_path=CONFIG['file_path'],
        store_id=CONFIG['store_id'],
        item_id=CONFIG['item_id'],
        window_size=CONFIG['window_size'],
        batch_size=CONFIG['batch_size']
    )

    # 3. Analyze Logically: Initialize Model
    model = SalesLSTM(
        input_dim=1, 
        hidden_dim=CONFIG['hidden_dim'], 
        num_layers=CONFIG['num_layers']
    ).to(device)

    # 4. Explore Systematically: Optimizer and Loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 5. Validate Rigorously: Training Loop
    print(f"Beginning training for Store {CONFIG['store_id']}, Item {CONFIG['item_id']}...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=CONFIG['epochs'],
        device=device
    )

    print(f"--- Process Complete ---")
    print(f"Model saved to: {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()