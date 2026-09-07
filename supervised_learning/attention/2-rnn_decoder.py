#!/usr/bin/env python3
"""
Module defining the RNNDecoder class for machine translation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder class for sequence-to-sequence models with attention
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units=units)

    def call(self, x, s_prev, hidden_states):
        """
        Decodes the target sequence step using attention
        """
        # Get attention context vector and weights
        context, _ = self.attention(s_prev, hidden_states)

        # Embed the previous word: shape (batch, 1, embedding)
        x = self.embedding(x)

        # Concatenate context vector and embedded x in that order
        # context shape: (batch, units) -> expand to (batch, 1, units)
        context_expanded = tf.expand_dims(context, 1)
        x = tf.concat([context_expanded, x], axis=-1)

        # Pass through GRU
        outputs, state = self.gru(x)

        # Output dense layer: shape (batch, vocab)
        # outputs shape is (batch, 1, units), reshape/squeeze to (batch, units)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, state
