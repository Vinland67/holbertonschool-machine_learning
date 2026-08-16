#!/usr/bin/env python3
"""
Module to build the DenseNet-121 architecture as described in
Densely Connected Convolutional Networks
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks

    growth_rate: growth rate
    compression: compression factor

    Input data has shape (224, 224, 3).
    All convolutions are preceded by Batch Normalization and a
    ReLU activation.
    All weights use he normal initialization with seed=0.

    Returns: the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    nb_filters = 2 * growth_rate

    bn1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(
        filters=nb_filters, kernel_size=(7, 7), strides=(2, 2),
        padding='same', kernel_initializer=init)(act1)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    dense1, nb_filters = dense_block(pool1, nb_filters, growth_rate, 6)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), padding='same')(dense4)

    output = K.layers.Dense(
        units=1000, activation='softmax',
        kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)
    return model
