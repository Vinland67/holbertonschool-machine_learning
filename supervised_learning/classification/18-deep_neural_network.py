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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for l in range(1, self.__L + 1):
            self.__weights["W" + str(l)] = (
                np.random.randn(layers[l - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.__weights["b" + str(l)] = np.zeros((layers[l - 1], 1))
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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            A_prev = self.__cache["A" + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            self.__cache["A" + str(l)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache
