#!/usr/bin/env python3
"""
Module to calculate specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    confusion: numpy.ndarray of shape (classes, classes)

    Returns: numpy.ndarray of shape (classes,) containing specificity
    """
    total = np.sum(confusion)
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = total - tp - fp - fn
    return tn / (tn + fp)
