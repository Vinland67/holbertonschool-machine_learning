#!/usr/bin/env python3
"""
Contains the continuous posterior function using Beta distribution.
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability p
    falls within a specific range [p1, p2].

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        p1: lower bound on the range
        p2: upper bound on the range

    Returns:
        The posterior probability that p is within [p1, p2]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Beta distribution parameters: alpha = x + 1, beta = n - x + 1
    a = x + 1
    b = n - x + 1

    # Cumulative Distribution Function (CDF) for Beta distribution
    # is the Regularized Incomplete Beta Function: special.betainc(a, b, p)
    cdf_p2 = special.betainc(a, b, p2)
    cdf_p1 = special.betainc(a, b, p1)

    return cdf_p2 - cdf_p1
