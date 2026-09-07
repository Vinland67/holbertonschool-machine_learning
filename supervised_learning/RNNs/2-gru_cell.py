#!/usr/bin/env python3
"""
Module defining the GRUCell class representing a Gated Recurrent Unit
"""
import numpy as np


class GRUCell:
    """
    Class that represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor for GRUCell

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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

        # Update gate
        z_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wz) + self.bz)))

        # Reset gate
        r_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wr) + self.br)))

        # Intermediate hidden state
        r_h_prev = r_t * h_prev
        concat_reset = np.concatenate((r_h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output
        logits = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return h_next, y
