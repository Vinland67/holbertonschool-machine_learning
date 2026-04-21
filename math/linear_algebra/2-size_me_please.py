#!/usr/bin/env python3
"""Calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    shape = []
    # Nə qədər ki element siyahıdır, ölçüsünü götür və bir qat dərinə get
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape
