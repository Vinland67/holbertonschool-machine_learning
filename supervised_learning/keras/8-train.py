#!/usr/bin/env python3
"""
Module to train a Keras model and save the best iteration
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model with options for early stopping, learning rate decay,
    and saving the best model iteration based on validation loss
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

        if save_best and filepath is not None:
            callbacks.append(K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
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
