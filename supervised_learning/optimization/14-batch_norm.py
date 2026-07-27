#!/usr/bin/env python3
"""
Module to create a batch normalization layer for a neural network
in TensorFlow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in
    TensorFlow

    prev: activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function that should be used on the
        output of the layer

    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)

    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon)

    return activation(Z_norm)
