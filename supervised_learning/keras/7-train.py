#!/usr/bin/env python3
"""
Module to train a Keras model with early stopping and learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping
    and inverse time learning rate decay
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            ))

        if learning_rate_decay:
            def lr_schedule(epoch):
                """Calculates step-wise inverse time decay"""
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(K.callbacks.LearningRateScheduler(
                schedule=lr_schedule,
                verbose=1
            ))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
