#!/usr/bin/env python3
"""
Module defining the deep_rnn function for forward propagation
through a deep Recurrent Neural Network.
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length l used for
                   forward propagation
        X: numpy.ndarray of shape (t, m, i) containing data to be used
        h_0: numpy.ndarray of shape (l, m, h) containing initial hidden state

    Returns:
        H: numpy.ndarray containing all of the hidden states
           shape (t + 1, l, m, h)
        Y: numpy.ndarray containing all of the outputs
           shape (t, m, o)
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Allocate memory for hidden states (t + 1 timesteps)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    for step in range(t):
        x_step = X[step]
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]
            h_next, y = cell.forward(h_prev, x_step)
            H[step + 1, layer] = h_next
            x_step = h_next

            # Y massivin ölçüsünü dinamik tapmaq üçün ilk t-də yaradaq
            if step == 0 and layer == l - 1:
                o = y.shape[1]
                Y = np.zeros((t, m, o))

        Y[step] = y

    return H, Y
