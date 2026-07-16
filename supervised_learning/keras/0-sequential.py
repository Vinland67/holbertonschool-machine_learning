#!/usr/bin/env python3
"""
Module to build a sequential neural network using Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    model = K.Sequential()

    # İlk qatı əlavə edirik
    model.add(K.layers.Dense(
        layers[0],
        input_dim=nx,
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    ))

    # keep_prob < 1 və ümumi qat sayı 1-dən çoxdursa, Dropout əlavə edirik
    if keep_prob < 1 and len(layers) > 1:
        model.add(K.layers.Dropout(1 - keep_prob))

    # Digər qatları ardıcıl olaraq əlavə edirik
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        # Sonuncu qatdan əvvəl dropout əlavə edirik (əgər keep_prob < 1)
        if keep_prob < 1 and i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
