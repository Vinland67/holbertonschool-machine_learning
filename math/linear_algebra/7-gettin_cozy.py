#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    if axis == 0:
        # Sütun sayları eyni olmalıdır
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Yeni siyahı yaradırıq ki orijinala toxunmayaq
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Sətir sayları eyni olmalıdır
        if len(mat1) != len(mat2):
            return None
        # Hər sətri qarşı tərəfdəki sətirlə birləşdiririk
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
