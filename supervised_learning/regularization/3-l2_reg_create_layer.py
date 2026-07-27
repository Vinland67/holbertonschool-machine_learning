#!/usr/bin/env python3
"""
Module to create a neural network layer in TensorFlow that
includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes
    L2 regularization

    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function that should be used on the layer
    lambtha: L2 regularization parameter

    Returns: the output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n,
                                   activation=activation,
                                   kernel_initializer=init,
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)
    return layer(prev)
