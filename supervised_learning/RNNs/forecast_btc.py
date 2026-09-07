#!/usr/bin/env python3
"""
Builds, trains, and validates a Keras model for BTC forecasting.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def create_dataset(X, y, batch_size=64, is_training=True):
    """
    Creates a tf.data.Dataset object from X and y arrays.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(input_shape):
    """
    Builds an RNN/LSTM model architecture for time series forecasting.
    """
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    """
    Main function to load preprocessed data, train the model, and save it.
    """
    data = np.load('preprocessed_btc.npz')
    X, y = data['X'], data['y']

    # Train / Validation split (80% train, 20% val)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    train_dataset = create_dataset(X_train, y_train, batch_size=64, is_training=True)
    val_dataset = create_dataset(X_val, y_val, batch_size=64, is_training=False)

    model = build_model(input_shape=(X.shape[1], X.shape[2]))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('btc_model.h5', save_best_only=True, monitor='val_loss')
    ]

    model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
