#!/usr/bin/env python3
"""
Preprocessing raw BTC time series data for RNN modeling.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_btc_data(file_path):
    """
    Cleans, resamples to hourly, scales, and windows raw BTC CSV data.

    Args:
        file_path: path to the raw BTC CSV dataset

    Returns:
        X: numpy.ndarray of shape (N, 24, 1)
        y: numpy.ndarray of shape (N, 1)
        scaler: fitted MinMaxScaler object
    """
    df = pd.read_csv(file_path)

    # Convert Timestamp to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Resample to hourly data
    df_hourly = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })

    # Forward fill missing values (if any)
    df_hourly.ffill(inplace=True)
    df_hourly.bfill(inplace=True)

    # Use Close price for forecasting
    close_prices = df_hourly[['Close']].values

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create 24-hour window sequences
    window_size = 24
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


if __name__ == '__main__':
    X, y, scaler = preprocess_btc_data('coinbaseUSD_1-min_data_2012-01-01_to_2019-01-09.csv')
    np.savez('preprocessed_btc.npz', X=X, y=y)
