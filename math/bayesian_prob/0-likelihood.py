#!/usr/bin/env python3
"""
Contains the likelihood function for Bayesian probability.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given
    various hypothetical probabilities.

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities

    Returns:
        1D numpy.ndarray containing the likelihood for each probability in P
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, (int, np.integer)) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Faktorial hesablama
    def fact(n):
        res = 1
        for i in range(1, n + 1):
            res *= i
        return res

    # Binomial Coefficient: n! / (x! * (n-x)!)
    combination = fact(n) / (fact(x) * fact(n - x))

    # Likelihood formula: P(data|hypo) = (nCx) * (p^x) * ((1-p)^(n-x))
    return combination * (P ** x) * ((1 - P) ** (n - x))
