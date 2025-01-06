import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to suppress warnings

import streamlit as st
import yfinance as yf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# App title
st.title("🎈 NVIDIA RNN Model Builder with Financial Data")
st.write("This app uses NVIDIA's stock data to build and customize an RNN!")

# Sidebar: User inputs for RNN configuration
st.sidebar.header("🔧 RNN Configuration")

# Number of RNN layers
num_layers = st.sidebar.slider("Number of RNN Layers", min_value=1, max_value=5, value=2, step=1)

# Units per layer
units_per_layer = []
for i in range(num_layers):
    units = st.sidebar.slider(f"Units in Layer {i+1}", min_value=1, max_value=256, value=64, step=1, key=f"units_{i}")
    units_per_layer.append(units)

# Activation function
activation_function = st.sidebar.selectbox(
    "Activation Function", ["relu", "tanh", "sigmoid"], index=1
)

# Output layer units
output_units = st.sidebar.slider("Units in Output Layer", min_value=1, max_value=10, value=1, step=1)

# Compile options
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)
loss_function = st.sidebar.selectbox("Loss Function", ["mse", "mae", "binary_crossentropy"], index=0)

# Section to fetch NVIDIA financial data
st.header("📊 NVIDIA Financial Data")
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
interval = st.selectbox("Select data interval:", ["1d", "1wk", "1mo"], index=0)

if st.button("Fetch NVIDIA Data"):
    try:
        # Fetch NVIDIA data using yfinance
        data = yf.download("NVDA", period=period, interval=interval)
        if not data.empty:
            st.write("### Historical Data for NVIDIA (NVDA)")
            st.dataframe(data)

            # Preprocess the data
            st.write("### Preprocessing Data")
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data[['Close']])
            
            # Prepare sequences for RNN
            sequence_length = st.slider("Sequence Length for RNN", min_value=5, max_value=50, value=20)
            X, y = [], []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i, 0])
                y.append(data_scaled[i, 0])
            X, y = np.array(X), np.array(y)

            # Reshape X for RNN input
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            st.write("### Processed Dataset")
            st.write(f"Input Shape: {X.shape}")
            st.write(f"Output Shape: {y.shape}")
        else:
            st.error("No data found for NVIDIA for the selected period and interval.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Button to trigger model building
if st.sidebar.button("Build RNN"):
    st.write("## 🔨 Building Your RNN Model...")

    # Dynamically build the RNN model
    model = Sequential()
    for i, units in enumerate(units_per_layer):
        if i == 0:
            # First layer needs input shape
            model.add(SimpleRNN(units, activation=activation_function, input_shape=(sequence_length, 1), return_sequences=(i < num_layers - 1)))
        else:
            # Subsequent layers
            model.add(SimpleRNN(units, activation=activation_function, return_sequences=(i < num_layers - 1)))

    # Add output layer
    model.add(Dense(output_units, activation="linear"))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_function)

    # Show the model summary
    st.write("### Model Summary")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    st.write("### Ready to Train!")
    st.write("You can now use the preprocessed dataset to train the RNN.")
