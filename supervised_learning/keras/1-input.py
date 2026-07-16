#!/usr/bin/env python3
"""
Module to build a neural network using Keras Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library (Functional API)
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    # İlk qatı əlavə edirik
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(x)

    # 1-dən çox qat varsa və keep_prob < 1-dirsə, Dropout əlavə edirik
    if keep_prob < 1 and len(layers) > 1:
        x = K.layers.Dropout(1 - keep_prob)(x)

    # Digər qatları ardıcıl olaraq əlavə edirik
    for i in range(1, len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        # Sonuncu qatdan əvvəl dropout əlavə edirik (əgər keep_prob < 1)
        if keep_prob < 1 and i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
