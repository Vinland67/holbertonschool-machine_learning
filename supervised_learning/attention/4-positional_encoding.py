#!/usr/bin/env python3
"""
Module defining the positional_encoding function for transformers
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len: maximum sequence length
        dm: model depth

    Returns:
        numpy.ndarray of shape (max_seq_len, dm)
        containing positional encodings
    """
    PE = np.zeros((max_seq_len, dm))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    div_term = np.power(
        10000, (2 * (i // 2)) / np.float32(dm)
    )
    angle_rates = pos / div_term

    PE[:, 0::2] = np.sin(angle_rates[:, 0::2])
    PE[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return PE
