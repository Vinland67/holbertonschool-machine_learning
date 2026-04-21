#!/usr/bin/env python3
"""Returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix"""
    # Matrisin hər bir sütunu üçün yeni bir sətir yaradırıq
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
