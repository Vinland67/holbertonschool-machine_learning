#!/usr/bin/env python3
"""
Module defining the RNNCell class representing a simple RNN cell
"""
import numpy as np


class RNNCell:
    """
    Class that represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor for RNNCell

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing
                    previous hidden state
            x_t: numpy.ndarray of shape (m, i) containing
                 data input for cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        logits = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return h_next, y
