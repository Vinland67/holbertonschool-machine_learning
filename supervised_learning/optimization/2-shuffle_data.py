#!/usr/bin/env python3
"""
Module to shuffle data points in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    X: numpy.ndarray of shape (m, nx) to shuffle
    Y: numpy.ndarray of shape (m, ny) to shuffle

    Returns: the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
