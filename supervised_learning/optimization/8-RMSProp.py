#!/usr/bin/env python3
"""
Module to set up the RMSProp optimization algorithm in TensorFlow
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow

    alpha: learning rate
    beta2: RMSProp weight (Discounting factor)
    epsilon: small number to avoid division by zero

    Returns: optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                             rho=beta2,
                                             epsilon=epsilon)
    return optimizer
