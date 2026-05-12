#!/usr/bin/env python3
"""
Contains the posterior function for Bayesian probability.
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various
    hypothetical probabilities.

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        1D numpy.ndarray containing the posterior probability for each P
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        msg = "Pr must be a numpy.ndarray with the same shape as P"
        raise TypeError(msg)
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Faktorial hesablama
    def fact(n):
        res = 1
        for i in range(1, n + 1):
            res *= i
        return res

    # 1. Likelihood
    num = fact(n)
    den = fact(x) * fact(n - x)
    combination = num / den
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # 2. Intersection (Likelihood * Prior)
    intersect = likelihood * Pr

    # 3. Marginal (Sum of Intersections)
    marginal = np.sum(intersect)

    # 4. Posterior (Intersection / Marginal)
    return intersect / marginal
