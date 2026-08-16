#!/usr/bin/env python3
"""
Module to build an identity block as described in
Deep Residual Learning for Image Recognition (2015)
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015)

    A_prev: output from the previous layer
    filters: tuple or list containing F11, F3, F12
        F11: number of filters in the first 1x1 convolution
        F3: number of filters in the 3x3 convolution
        F12: number of filters in the second 1x1 convolution

    All convolutions are followed by batch normalization along
    the channels axis and a ReLU activation, except the last
    which is followed by batch normalization only (before the
    addition and final ReLU).
    All weights use he normal initialization with seed=0.

    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1), padding='same',
        kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), padding='same',
        kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), padding='same',
        kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    add = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
