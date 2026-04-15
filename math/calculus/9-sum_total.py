#!/usr/bin/env python3
"""Summation of i squared"""


def summation_i_squared(n):
    """Calculates the sum of i^2 from 1 to n"""
    if not isinstance(n, int) or n < 1:
        return None
    total = (n * (n + 1) * (2 * n + 1)) // 6
    return total
