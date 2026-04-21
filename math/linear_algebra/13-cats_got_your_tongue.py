#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis using NumPy"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    return np.concatenate((mat1, mat2), axis=axis)
