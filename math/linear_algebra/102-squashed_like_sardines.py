#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis recursively"""


def cat_matrices(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    # Matrislərin ölçülərini (shape) yoxlamaq üçün köməkçi funksiya
    def get_shape(matrix):
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            if len(matrix) == 0:
                break
            matrix = matrix[0]
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Əgər ox (axis) matrisin ölçüsündən böyükdürsə
    if axis >= len(shape1) or axis >= len(shape2):
        return None

    # Birləşdirilən oxdan başqa digər bütün ölçülər eyni olmalıdır
    for i in range(len(shape1)):
        if i != axis:
            if i >= len(shape2) or shape1[i] != shape2[i]:
                return None

    # Rekursiv birləşdirmə
    def recursive_cat(m1, m2, cur_axis):
        if cur_axis == 0:
            return m1 + m2

        res = []
        for i in range(len(m1)):
            merged = recursive_cat(m1[i], m2[i], cur_axis - 1)
            if merged is None:
                return None
            res.append(merged)
        return res

    return recursive_cat(mat1, mat2, axis)
