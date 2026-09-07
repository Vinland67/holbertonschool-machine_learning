#!/usr/bin/env python3
"""
Module defining the sdp_attention function for scaled dot-product attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot-product attention

    Args:
        Q: query tensor of shape (..., seq_len_q, dk)
        K: key tensor of shape (..., seq_len_v, dk)
        V: value tensor of shape (..., seq_len_v, dv)
        mask: optional mask tensor that can be broadcast
              into (..., seq_len_q, seq_len_v)

    Returns:
        output, weights
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
