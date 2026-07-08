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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for idx in range(1, self.__L + 1):
            self.__weights["W" + str(idx)] = (
                np.random.randn(layers[idx - 1], prev) *
                np.sqrt(2 / prev)
            )
            self.__weights["b" + str(idx)] = np.zeros((layers[idx - 1], 1))
            prev = layers[idx - 1]

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
        for idx in range(1, self.__L + 1):
            W = self.__weights["W" + str(idx)]
            b = self.__weights["b" + str(idx)]
            A_prev = self.__cache["A" + str(idx - 1)]
            Z = np.dot(W, A_prev) + b
            self.__cache["A" + str(idx)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y

        for idx in range(self.__L, 0, -1):
            A_prev = cache["A" + str(idx - 1)]
            W = self.__weights["W" + str(idx)]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if idx > 1:
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights["W" + str(idx)] -= alpha * dW
            self.__weights["b" + str(idx)] -= alpha * db
