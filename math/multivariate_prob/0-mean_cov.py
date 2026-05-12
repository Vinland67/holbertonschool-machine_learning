#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set.
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set

    Returns:
        mean: numpy.ndarray of shape (1, d)
        cov: numpy.ndarray of shape (d, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean hesablama (axis=0 sütunlar üzrə deməkdir)
    mean = np.mean(X, axis=0, keepdims=True)

    # Data mərkəzləşdirmə (Broadcasting istifadə olunur)
    X_centered = X - mean

    # Covariance: (X_centered.T @ X_centered) / (n - 1)
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
