#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    # Vurulma şərtini yoxlayırıq: mat1 columns == mat2 rows
    if len(mat1[0]) != len(mat2):
        return None

    # Yeni matris yaradırıq (ölçüsü: mat1_rows x mat2_cols)
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Element-wise vurulma və cəmləmə (dot product)
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            row.append(total)
        result.append(row)

    return result
