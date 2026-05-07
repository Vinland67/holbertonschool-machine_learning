#!/usr/bin/env python3
"""Contains the Binomial class to represent a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            sum_diff_sq = sum((x - mean) ** 2 for x in data)
            variance = sum_diff_sq / len(data)
            p_estimated = 1 - (variance / mean)
            self.n = int(round(mean / p_estimated))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Faktorial funksiyası (internal helper)
        def fact(n):
            res = 1
            for i in range(1, n + 1):
                res *= i
            return res

        # Kombinasiya: n! / (k! * (n-k)!)
        n_fact = fact(self.n)
        k_fact = fact(k)
        nk_fact = fact(self.n - k)
        combination = n_fact / (k_fact * nk_fact)

        # PMF = combination * (p^k) * ((1-p)^(n-k))
        pmf_val = combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_val
