#!/usr/bin/env python3
"""
Module to determine if gradient descent should be stopped early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early

    cost: current validation cost of the neural network
    opt_cost: lowest recorded validation cost of the neural network
    threshold: threshold used for early stopping
    patience: patience count used for early stopping
    count: count of how long the threshold has not been met

    Returns: a boolean of whether the network should be stopped
        early, followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop = count >= patience
    return stop, count
