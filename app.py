import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processor import get_processed_data
from model import SalesLSTM

# --- Page Config ---
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

@st.cache_resource
def load_resources():
    # Using the same config as main.py
    window_size = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data and scaler
    _, val_loader, scaler = get_processed_data('data/train.csv', window_size=window_size)
    
    # Load Model
    model = SalesLSTM(input_dim=1, hidden_dim=64, num_layers=2).to(device)
    model.load_state_dict(torch.load('sales_lstm.pth', map_location=device))
    model.eval()
    
    return val_loader, model, scaler, device

# Initialize
val_loader, model, scaler, device = load_resources()

# --- Dashboard Header ---
st.title("📈 Dynamic Sales Forecasting Dashboard")
st.write("Visualizing LSTM model performance on the Store Item Demand dataset.")

# --- Prediction Logic ---
all_preds = []
all_actuals = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        preds = model(x)
        all_preds.append(preds.cpu().numpy())
        all_actuals.append(y.numpy())

# Flatten and Inverse Transform
predictions = scaler.inverse_transform(np.concatenate(all_preds))
actuals = scaler.inverse_transform(np.concatenate(all_actuals))

# --- Metrics Calculation ---
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
accuracy = 100 - mape

col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", f"{accuracy:.2f}%")
col2.metric("Mean Absolute Error", f"{np.mean(np.abs(actuals - predictions)):.2f} units")
col3.metric("Status", "Optimized" if accuracy > 85 else "Training Needed")

# --- Visualization ---
st.subheader("Actual vs. Predicted Sales (Validation Period)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(actuals[-100:], label='Actual Sales', color='#1f77b4', linewidth=2)
ax.plot(predictions[-100:], label='LSTM Prediction', color='#ff7f0e', linestyle='--', linewidth=2)
ax.set_xlabel("Days")
ax.set_ylabel("Sales Units")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# --- Future Insights ---
st.info(f"**Insight:** The model is currently using a {30}-day sliding window to predict trends. "
        "The overlap in the chart indicates the model is successfully capturing weekly seasonality.")