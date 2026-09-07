#!/usr/bin/env python3
"""
Module defining the MultiHeadAttention class for transformers
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention layer class
    """

    def __init__(self, dm, h):
        """
        Class constructor
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result.
        """
        x = tf.reshape(
            x, (batch_size, -1, self.h, self.depth)
        )
        return tf.transpose(
            x, perm=[0, 2, 1, 3]
        )

    def call(self, Q, K, V, mask):
        """
        Performs multi head attention forward pass
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        scaled_attention, weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )

        output = self.linear(concat_attention)

        return output, weights
