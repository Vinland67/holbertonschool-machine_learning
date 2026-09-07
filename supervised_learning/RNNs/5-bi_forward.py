#!/usr/bin/env python3
"""
Module defining the BidirectionalCell class for a bidirectional RNN
"""
import numpy as np


class BidirectionalCell:
    """
    Class that represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor for BidirectionalCell

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) containing data input
                 for the cell

        Returns:
            h_next: next hidden state
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat_input, self.Whf) + self.bhf)

        return h_next
