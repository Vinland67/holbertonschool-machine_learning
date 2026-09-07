#!/usr/bin/env python3
"""Module to convert a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Converts a Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: trained Gensim Word2Vec model.

    Returns:
        The trainable Keras Embedding layer.
    """
    weights = model.wv.vectors

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )

    return layer
