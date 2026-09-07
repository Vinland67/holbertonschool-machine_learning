#!/usr/bin/env python3
"""
Module defining the LSTMCell class representing an LSTM unit
"""
import numpy as np


class LSTMCell:
    """
    Class that represents an LSTM unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor for LSTMCell

        Args:
            i: dimensionality of the data input
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing
                    previous hidden state
            c_prev: numpy.ndarray of shape (m, h) containing
                    previous cell state
            x_t: numpy.ndarray of shape (m, i) containing
                 data input for cell

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wf) + self.bf)))

        # Update gate
        u_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wu) + self.bu)))

        # Candidate cell state
        c_tilde = np.tanh(np.matmul(concat_input, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wo) + self.bo)))

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Output
        logits = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        return h_next, c_next, y
