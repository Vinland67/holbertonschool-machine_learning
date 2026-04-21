#!/usr/bin/env python3
"""Slices a matrix along specific axes dynamically"""


def np_slice(matrix, axes={}):
    """Function that slices a matrix along specific axes"""
    # Bütün oxlar üçün "hamısını götür" (:) slice-ı yaradırıq
    slices = [slice(None)] * matrix.ndim

    # Verilən oxlar üzrə slice obyektlərini yeniləyirik
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)

    # Matrisi həmin slice siyahısı ilə kəsirik
    return matrix[tuple(slices)]
