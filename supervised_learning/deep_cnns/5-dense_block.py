#!/usr/bin/env python3
"""
Module to build a dense block as described in
Densely Connected Convolutional Networks
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks

    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs
    """
    init = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        # Bottleneck layer (DenseNet-B)
        bn1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(act1)

        # Standard 3x3 Conv layer
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(act2)

        # Concatenate input with current output
        X = K.layers.Concatenate(axis=3)([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
