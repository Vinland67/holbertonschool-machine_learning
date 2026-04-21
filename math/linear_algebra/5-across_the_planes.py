#!/usr/bin/env python3
"""Adds two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two 2D matrices element-wise"""
    # Əvvəlcə sətir saylarını yoxlayırıq
    if len(mat1) != len(mat2):
        return None
    
    # Sonra hər bir sətrin (sütunların) uzunluğunu yoxlayırıq
    if len(mat1[0]) != len(mat2[0]):
        return None

    # İç-içə list comprehension ilə toplama
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
