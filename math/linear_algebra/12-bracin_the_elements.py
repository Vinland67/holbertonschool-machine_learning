#!/usr/bin/env python3
"""Performs element-wise addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """Function that performs element-wise operations using NumPy"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
