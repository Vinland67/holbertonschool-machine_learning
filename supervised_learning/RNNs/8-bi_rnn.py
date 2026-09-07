#!/usr/bin/env python3
"""
Module defining the bi_rnn function for forward propagation
through a bidirectional RNN.
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell used for forward propagation
        X: numpy.ndarray of shape (t, m, i) containing the input data
        h_0: numpy.ndarray of shape (m, h) containing the initial hidden
             state in the forward direction
        h_t: numpy.ndarray of shape (m, h) containing the initial hidden
             state in the backward direction

    Returns:
        H: numpy.ndarray containing all of the concatenated hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    h_prev = h_0
    h_next = h_t

    for step in range(t):
        # Forward pass (0-dan t-1-ə)
        h_prev = bi_cell.forward(h_prev, X[step])
        H_f[step] = h_prev

        # Backward pass (t-1-dən 0-a)
        back_step = t - 1 - step
        h_next = bi_cell.backward(h_next, X[back_step])
        H_b[back_step] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_f, H_b), axis=2)

    # Calculate outputs using the cell's output method
    Y = bi_cell.output(H)

    return H, Y
