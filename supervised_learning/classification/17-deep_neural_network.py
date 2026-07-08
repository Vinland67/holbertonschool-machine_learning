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
        for n in layers:
            if not isinstance(n, int) or n < 1:
                raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for l in range(1, self.__L + 1):
            self.__weights[f'W{l}'] = (
                np.random.randn(layers[l - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.__weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
            prev = layers[l - 1]

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
