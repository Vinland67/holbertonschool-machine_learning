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

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Args:
            h_next: numpy.ndarray of shape (m, h) containing next
                    hidden state
            x_t: numpy.ndarray of shape (m, i) containing data input
                 for the cell

        Returns:
            h_pev: previous hidden state
        """
        concat_input = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.matmul(concat_input, self.Whb) + self.bhb)

        return h_pev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) containing the
               concatenated hidden states from both directions, excluding
               their initialized states

        Returns:
            Y: the outputs of shape (t, m, o)
        """
        t, m, _ = H.shape
        o = self.Wy.shape[1]

        Y = np.zeros((t, m, o))

        for step in range(t):
            logits = np.matmul(H[step], self.Wy) + self.by
            exp_logits = np.exp(logits)
            Y[step] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return Y
