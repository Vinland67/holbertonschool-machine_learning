#!/usr/bin/env python3
"""
Module to calculate precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    confusion: numpy.ndarray of shape (classes, classes)

    Returns: numpy.ndarray of shape (classes,) containing precision
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
