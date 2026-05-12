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

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = np.matmul(X_centered, X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point x.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF at x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Det və Inv hesablayırıq
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Məxrəc: sqrt((2 * pi)^d * det)
        denom = np.sqrt(((2 * np.pi) ** d) * det)

        # Üst (Exponent) hissəsi: -0.5 * (x - mu).T @ inv @ (x - mu)
        diff = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)

        # PDF = (1 / denom) * exp(exponent)
        # Nəticə 1x1 matris olduğu üçün item() ilə skalyar alırıq
        pdf_val = (1 / denom) * np.exp(exponent)

        return pdf_val.item()
