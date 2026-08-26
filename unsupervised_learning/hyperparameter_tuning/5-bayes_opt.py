#!/usr/bin/env python3
"""
Module defining the BayesianOptimization class with optimize method
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that performs Bayesian optimization on a
    noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor for BayesianOptimization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_opt = np.min(self.gp.Y)
            improvement = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            improvement = mu - y_opt - self.xsi

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / sigma

            # Standard normal PDF
            pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (Z ** 2))

            # Standard normal CDF approximation (no math or scipy allowed)
            t = 1.0 / (1.0 + 0.2316419 * np.abs(Z))
            b1 = 0.319381530
            b2 = -0.356563782
            b3 = 1.781477937
            b4 = -1.821255978
            b5 = 1.330274429
            prob = pdf * (b1*t + b2*(t**2) + b3*(t**3) + b4*(t**4) + b5*(t**5))
            cdf = np.where(Z >= 0, 1.0 - prob, prob)

            ei = improvement * cdf + sigma * pdf
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Precision check using np.isclose to stop early correctly
            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
