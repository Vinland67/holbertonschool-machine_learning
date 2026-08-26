#!/usr/bin/env python3
"""
Module defining the BayesianOptimization class with optimize method
"""
import numpy as np
from scipy.stats import norm
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

        with np.errstate(divide='ignore'):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Args:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing optimal point
            Y_opt: numpy.ndarray of shape (1,) representing optimal value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
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
