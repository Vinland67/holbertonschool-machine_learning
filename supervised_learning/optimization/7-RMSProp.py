#!/usr/bin/env python3
"""
Module to update a variable using the RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    alpha: learning rate
    beta2: RMSProp weight
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    s: previous second moment of var

    Returns: the updated variable and the new moment, respectively
    """
    s_new = beta2 * s + (1 - beta2) * grad ** 2
    var_new = var - alpha * grad / ((s_new ** 0.5) + epsilon)
    return var_new, s_new
