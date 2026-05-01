#!/usr/bin/env python3
"""Calculates the inverse of a matrix without any imports"""


def determinant(matrix):
    """Helper function to calculate the determinant of a matrix"""
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)
    return det


def cofactor(matrix):
    """Helper function to calculate the cofactor matrix"""
    if len(matrix) == 1:
        return [[1]]

    cofactor_mat = []
    for r in range(len(matrix)):
        cofactor_row = []
        for c in range(len(matrix)):
            sub_matrix = [row[:c] + row[c + 1:] for row in
                          (matrix[:r] + matrix[r + 1:])]
            min_val = determinant(sub_matrix)
            cofactor_row.append(((-1) ** (r + c)) * min_val)
        cofactor_mat.append(cofactor_row)

    return cofactor_mat


def adjugate(matrix):
    """Helper function to calculate the adjugate matrix"""
    cof_mat = cofactor(matrix)
    adj_mat = []
    for c in range(len(cof_mat[0])):
        adj_row = []
        for r in range(len(cof_mat)):
            adj_row.append(cof_mat[r][c])
        adj_mat.append(adj_row)
    return adj_mat


def inverse(matrix):
    """Calculates the inverse of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    if len(matrix) == 1:
        return [[1 / det]]

    adj = adjugate(matrix)

    inv_mat = []
    for r in range(len(adj)):
        inv_row = []
        for c in range(len(adj[0])):
            inv_row.append(adj[r][c] / det)
        inv_mat.append(inv_row)

    return inv_mat
