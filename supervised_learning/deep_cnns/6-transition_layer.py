#!/usr/bin/env python3
"""
Module to build a transition layer as described in
Densely Connected Convolutional Networks
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks

    X: output from the previous layer
    nb_filters: integer representing the number of filters in X
    compression: compression factor for the transition layer

    Implements compression as used in DenseNet-C.
    All weights use he normal initialization with seed=0.
    All convolutions are preceded by Batch Normalization and a
    ReLU activation.

    Returns: the output of the transition layer and the number
        of filters within the output, respectively
    """
    init = K.initializers.HeNormal(seed=0)

    nb_filters = int(nb_filters * compression)

    bn = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(bn)

    conv = K.layers.Conv2D(
        filters=nb_filters, kernel_size=(1, 1), padding='same',
        kernel_initializer=init)(act)

    pool = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='same')(conv)

    return pool, nb_filters
