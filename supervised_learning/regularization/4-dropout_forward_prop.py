#!/usr/bin/env python3
"""
Module to conduct forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    X: numpy.ndarray of shape (nx, m) containing the input data
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the network
    keep_prob: probability that a node will be kept

    All layers except the last use tanh activation; the last layer
    uses softmax activation.

    Returns: a dictionary containing the outputs of each layer and
        the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            cache['A' + str(i)] = A
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
