#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    labels: one-hot numpy.ndarray of shape (m, classes) - correct labels
    logits: one-hot numpy.ndarray of shape (m, classes) - predicted labels

    Returns: confusion numpy.ndarray of shape (classes, classes)
    """
    return np.dot(labels.T, logits)
