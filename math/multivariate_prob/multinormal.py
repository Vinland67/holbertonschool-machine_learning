#!/usr/bin/env python3
"""
Contains the MultiNormal class that represents a
Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor for MultiNormal.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # d = dimensions, n = number of data points
        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean hesablama: Hər dimensiya üçün (sətirlər üzrə, axis=1)
        # Formanın (d, 1) olması üçün keepdims=True istifadə edirik
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Data mərkəzləşdirmə
        X_centered = data - self.mean

        # Covariance hesablama: (d, n) * (n, d) -> (d, d)
        # Düstur: (1 / (n - 1)) * (X_centered @ X_centered.T)
        self.cov = np.matmul(X_centered, X_centered.T) / (n - 1)
