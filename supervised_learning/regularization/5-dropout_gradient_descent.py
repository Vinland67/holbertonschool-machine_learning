#!/usr/bin/env python3
"""
Module to update weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent

    Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs and dropout masks of each layer
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network

    All layers use tanh activation except the last (softmax).
    Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = (dA_prev * cache['D' + str(i - 1)]) / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
