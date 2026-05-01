#!/usr/bin/env python3
"""Calculates the determinant of a matrix recursively without any imports"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    # Validasiya 1: Siyahıların siyahısı olub-olmaması
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matris halı [[]]
    if matrix == [[]]:
        return 1

    # Validasiya 2: Kvadrat matris olub-olmaması
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    # 1x1 matris halı
    if len(matrix) == 1:
        return matrix[0][0]

    # 2x2 matris halı
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # NxN matris üçün rekursiv determinant hesablama
    det = 0
    for col in range(len(matrix)):
        # Hər bir elementin minorunu tapırıq
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        # İşarəni təyin edirik (-1)^col
        sign = (-1) ** col
        # Rekursiv çağırış
        det += sign * matrix[0][col] * determinant(minor)

    return det
