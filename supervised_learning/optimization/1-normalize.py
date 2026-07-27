#!/usr/bin/env python3
"""
Module to normalize (standardize) a matrix
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    X: numpy.ndarray of shape (d, nx) to normalize
    m: numpy.ndarray of shape (nx,) containing the mean of all
        features of X
    s: numpy.ndarray of shape (nx,) containing the standard deviation
        of all features of X

    Returns: The normalized X matrix
    """
    return (X - m) / s
