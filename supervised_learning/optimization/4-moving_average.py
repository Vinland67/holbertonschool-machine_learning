#!/usr/bin/env python3
"""
Module to calculate the weighted moving average of a data set
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set,
    using bias correction

    data: list of data to calculate the moving average of
    beta: weight used for the moving average

    Returns: a list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for i, value in enumerate(data):
        v = beta * v + (1 - beta) * value
        corrected_v = v / (1 - beta ** (i + 1))
        moving_averages.append(corrected_v)

    return moving_averages
