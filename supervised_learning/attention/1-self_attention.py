#!/usr/bin/env python3
"""
Module defining the SelfAttention class for sequence-to-sequence models
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self Attention class for alignment model in machine translation
    """

    def __init__(self, units):
        """
        Class constructor
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Calculates the attention context vector and weights
        """
        # s_prev shape: (batch, units) -> expand to (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # W(s_prev) shape: (batch, 1, units)
        # U(hidden_states) shape: (batch, input_seq_len, units)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))

        # weights shape: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # context shape: (batch, units)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
