#!/usr/bin/env python3
"""Module that defines a deep neural network with multiple activations"""
import os
import pickle
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
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

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
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

    @property
    def activation(self):
        """Getter for activation"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for idx in range(1, self.__L + 1):
            W = self.__weights["W" + str(idx)]
            b = self.__weights["b" + str(idx)]
            A_prev = self.__cache["A" + str(idx - 1)]
            Z = np.dot(W, A_prev) + b
            if idx == self.__L:
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache["A" + str(idx)] = t / np.sum(t, axis=0,
                                                          keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache["A" + str(idx)] = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    self.__cache["A" + str(idx)] = np.tanh(Z)
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using multiclass cross-entropy"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
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
                if self.__activation == 'sig':
                    dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
                elif self.__activation == 'tanh':
                    dZ = np.dot(W.T, dZ) * (1 - (A_prev ** 2))

            self.__weights["W" + str(idx)] -= alpha * dW
            self.__weights["b" + str(idx)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps = []
        costs = []

        for idx in range(iterations + 1):
            if idx > 0:
                self.gradient_descent(Y, self.__cache, alpha)

            A, _ = self.forward_prop(X)

            if idx == 0 or idx == iterations or idx % step == 0:
                cost = self.cost(Y, A)
                steps.append(idx)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(idx, cost))

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.exists(filename) and not filename.endswith('.pkl'):
            filename += '.pkl'
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
