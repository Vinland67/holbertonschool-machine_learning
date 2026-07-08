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

        types = set(map(type, layers))
        if types != {int} or min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for idx in range(1, self.L + 1):
            self.weights["W" + str(idx)] = (
                np.random.randn(layers[idx - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.weights["b" + str(idx)] = np.zeros((layers[idx - 1], 1))
            prev = layers[idx - 1]
