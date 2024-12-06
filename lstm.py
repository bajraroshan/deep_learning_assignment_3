import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random

# Set random seed for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    """
    Load the dataset, preprocess it, and create moving averages and lag features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure the dataset is available.")
    
    data = pd.read_csv(file_path)
    if 'Date' not in data.columns or 'Close' not in data.columns:
        raise ValueError("Dataset must contain 'Date' and 'Close' columns.")
    
    # Set 'Date' as the index and preprocess numerical columns
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '').astype(float)
    
    # Create moving averages
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    
    # Create lag features
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['Lag3'] = data['Close'].shift(3)
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

# Function to split data into train and test sets
def split_data(data, features, target, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    train = train.reset_index()
    test = test.reset_index()
    train_X = train[features].values
    train_y = train[target].values
    test_X = test[features].values
    test_y = test[target].values
    return train, test, train_X, train_y, test_X, test_y

# Function to scale and reshape data for LSTM
def scale_and_reshape(train_X, train_y, test_X, test_y):
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1))
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, 1))
    train_X_scaled = train_X_scaled.reshape((train_X_scaled.shape[0], 1, train_X_scaled.shape[1]))
    test_X_scaled = test_X_scaled.reshape((test_X_scaled.shape[0], 1, test_X_scaled.shape[1]))
    return train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, scaler_y

# Function to build the LSTM model
def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to calculate MSE, RMSE, and MAPE
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, rmse, mape

# Function to plot actual vs predicted prices
def plot_actual_vs_predicted(train, test, train_y, test_y, train_predict, test_predict, title="Actual vs Predicted Prices"):
    plt.figure(figsize=(16, 8))
    plt.plot(train['Date'], train_y, label='Train Actual', color='blue')
    plt.plot(test['Date'], test_y, label='Test Actual', color='green')
    plt.plot(train['Date'], train_predict, label='Train Predicted', linestyle='--', color='orange')
    plt.plot(test['Date'], test_predict, label='Test Predicted', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.show()

# Function to plot moving averages
def plot_moving_averages(data, title="Moving Averages"):
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['MA7'], label='7-Day MA', linestyle='--', color='orange')
    plt.plot(data.index, data['MA30'], label='30-Day MA', linestyle='--', color='green')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.show()

# Function to plot lag features
def plot_lag_features(data, title="Lag Features"):
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['Lag1'], label='Lag 1', linestyle='--', color='orange')
    plt.plot(data.index, data['Lag2'], label='Lag 2', linestyle='--', color='green')
    plt.plot(data.index, data['Lag3'], label='Lag 3', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = 'Google_Stock_Price_Train.csv'
    features = ['Close', 'MA7', 'MA30', 'Lag1', 'Lag2', 'Lag3']
    target = 'Close'

    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    train, test, train_X, train_y, test_X, test_y = split_data(data, features, target)
    train_X_scaled, train_y_scaled, test_X_scaled, test_y_scaled, scaler_y = scale_and_reshape(train_X, train_y, test_X, test_y)

    # Build and train LSTM model
    model = build_lstm_model((1, len(features)))
    history = model.fit(train_X_scaled, train_y_scaled, batch_size=1, epochs=30)

    # Make predictions
    train_predict_scaled = model.predict(train_X_scaled)
    test_predict_scaled = model.predict(test_X_scaled)
    train_predict = scaler_y.inverse_transform(train_predict_scaled)
    test_predict = scaler_y.inverse_transform(test_predict_scaled)

    # Metrics
    train_mse, train_rmse, train_mape = calculate_metrics(train_y, train_predict)
    test_mse, test_rmse, test_mape = calculate_metrics(test_y, test_predict)
    print(f'Train MSE: {train_mse}, RMSE: {train_rmse}, MAPE: {train_mape}%')
    print(f'Test MSE: {test_mse}, RMSE: {test_rmse}, MAPE: {test_mape}%')

    # Plot results
    plot_actual_vs_predicted(train, test, train_y, test_y, train_predict, test_predict)
    plot_moving_averages(data)
    plot_lag_features(data)
