#!/usr/bin/env python3
"""Module that defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for l in range(1, self.L + 1):
            self.weights[f'W{l}'] = (
                np.random.randn(layers[l - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
            prev = layers[l - 1]
