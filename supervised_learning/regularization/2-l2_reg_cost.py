#!/usr/bin/env python3
"""
Module to calculate the cost of a neural network with L2 regularization
using Keras
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    cost: tensor containing the cost of the network without L2
        regularization
    model: Keras model that includes layers with L2 regularization

    Returns: a tensor containing the total cost for each layer of
        the network, accounting for L2 regularization
    """
    l2_costs = []
    for loss in model.losses:
        l2_costs.append(cost + loss)

    return tf.stack(l2_costs)
