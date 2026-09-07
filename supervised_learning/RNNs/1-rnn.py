#!/usr/bin/env python3
"""
Module defining the rnn function for simple RNN forward propagation
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Args:
        rnn_cell: instance of RNNCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) containing data input
        h_0: numpy.ndarray of shape (m, h) containing initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states, shape (t + 1, m, h)
        Y: numpy.ndarray containing all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Execute one forward step to get output dimension 'o'
    _, first_y = rnn_cell.forward(h_0, X[0])
    o = first_y.shape[1]

    Y = np.zeros((t, m, o))

    for time_step in range(t):
        h_next, y = rnn_cell.forward(H[time_step], X[time_step])
        H[time_step + 1] = h_next
        Y[time_step] = y

    return H, Y
