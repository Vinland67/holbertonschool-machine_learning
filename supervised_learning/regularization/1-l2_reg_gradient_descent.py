#!/usr/bin/env python3
"""
Module to update weights and biases using gradient descent with
L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs of each layer of the neural network
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers of the network

    Uses tanh activations on each layer except the last (softmax).
    Updates weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
