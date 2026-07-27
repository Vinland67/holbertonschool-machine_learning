#!/usr/bin/env python3
"""
Module to update a variable using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    alpha: learning rate
    beta1: weight used for the first moment
    beta2: weight used for the second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: previous first moment of var
    s: previous second moment of var
    t: time step used for bias correction

    Returns: the updated variable, the new first moment, and the
        new second moment, respectively
    """
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * grad ** 2

    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    var_new = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)

    return var_new, v_new, s_new
