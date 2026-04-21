#!/usr/bin/env python3
"""Adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    # Ölçüləri yoxlayırıq
    if len(arr1) != len(arr2):
        return None

    # Yeni siyahı yaradaraq elementləri toplayırıq
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
