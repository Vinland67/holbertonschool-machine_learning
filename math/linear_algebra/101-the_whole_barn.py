#!/usr/bin/env python3
"""Adds two matrices of any dimension recursively"""


def add_matrices(mat1, mat2):
    """Function that adds two matrices of any dimension"""
    # Ölçüləri yoxlayırıq (hər səviyyədə)
    if len(mat1) != len(mat2):
        return None

    # Əgər daxildəki element siyahı deyilsə (rəqəmdirsə), birbaşa toplayırıq
    if not isinstance(mat1[0], list):
        # Əgər mat2[0] siyahıdırsa, amma mat1[0] deyilsə - ölçü fərqlidir
        if isinstance(mat2[0], list):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    # Rekursiv olaraq hər bir alt-siyahını toplayırıq
    res = []
    for i in range(len(mat1)):
        inner_res = add_matrices(mat1[i], mat2[i])
        # Əgər alt-qatdan None qayıdıbsa, deməli ölçülər uyğun deyil
        if inner_res is None:
            return None
        res.append(inner_res)

    return res
