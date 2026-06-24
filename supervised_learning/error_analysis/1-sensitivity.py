#!/usr/bin/env python3
"""
Module to calculate sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    confusion: numpy.ndarray of shape (classes, classes)

    Returns: numpy.ndarray of shape (classes,) containing sensitivity
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
