import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense
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

# Function to scale and reshape data
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

# Function to build models
def build_model(model_type, input_shape):
    """
    Build and compile a model based on the type: Vanilla RNN, GRU, or LSTM.
    """
    model = Sequential()
    if model_type == "Vanilla RNN":
        model.add(SimpleRNN(50, return_sequences=True, input_shape=input_shape))
        model.add(SimpleRNN(50, return_sequences=False))
    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50, return_sequences=False))
    elif model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to calculate MSE, RMSE, and MAPE
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, rmse, mape

# Function to plot MSE, RMSE, and MAPE for all models
def plot_metrics(metrics, models, metric_labels):
    """
    Plot the metrics (MSE, RMSE, MAPE) for the given models.
    """
    metrics_array = np.array(metrics).T  # Transpose to align metrics for each model
    x = np.arange(len(models))  # Label positions
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    for i, metric_label in enumerate(metric_labels):
        ax[i].bar(x, metrics_array[i], color=['blue', 'orange', 'green'])
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(models)
        ax[i].set_title(f'{metric_label} Comparison')
        ax[i].set_ylabel(metric_label)
    
    plt.tight_layout()
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

    models = ["Vanilla RNN", "GRU", "LSTM"]
    metrics = []

    for model_type in models:
        # Build and train the model
        model = build_model(model_type, (1, len(features)))
        model.fit(train_X_scaled, train_y_scaled, batch_size=1, epochs=10, verbose=0)
        
        # Make predictions
        train_predict_scaled = model.predict(train_X_scaled)
        test_predict_scaled = model.predict(test_X_scaled)
        train_predict = scaler_y.inverse_transform(train_predict_scaled)
        test_predict = scaler_y.inverse_transform(test_predict_scaled)

        # Calculate metrics
        train_metrics = calculate_metrics(train_y, train_predict)
        test_metrics = calculate_metrics(test_y, test_predict)
        metrics.append(train_metrics + test_metrics)  # Combine Train and Test Metrics
    
    # Display Metrics
    metric_labels = ["MSE", "RMSE", "MAPE"]
    print("Metrics for All Models:")
    for model, metric in zip(models, metrics):
        print(f"\n{model}:")
        print(f"Train MSE: {metric[0]}, RMSE: {metric[1]}, MAPE: {metric[2]}%")
        print(f"Test MSE: {metric[3]}, RMSE: {metric[4]}, MAPE: {metric[5]}%")
    
    # Plot metrics
    plot_metrics(metrics, models, metric_labels)
