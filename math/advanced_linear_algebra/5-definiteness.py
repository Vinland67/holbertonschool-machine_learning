#!/usr/bin/env python3
"""Calculates the definiteness of a matrix using numpy"""
import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues, _ = np.linalg.eig(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigenvalues > 1e-10)
    pos_semi = (np.all(eigenvalues >= -1e-10) and
                np.any(np.isclose(eigenvalues, 0)))
    neg = np.all(eigenvalues < -1e-10)
    neg_semi = (np.all(eigenvalues <= 1e-10) and
                np.any(np.isclose(eigenvalues, 0)))

    if pos:
        return "Positive definite"
    if pos_semi:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi:
        return "Negative semi-definite"

    if np.any(eigenvalues > 1e-10) and np.any(eigenvalues < -1e-10):
        return "Indefinite"

    return None
