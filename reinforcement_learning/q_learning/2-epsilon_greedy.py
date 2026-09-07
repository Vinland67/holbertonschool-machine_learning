#!/usr/bin/env python3
"""
Module defining the epsilon_greedy function
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Args:
        Q: numpy.ndarray containing the Q-table
        state: current state
        epsilon: epsilon value to use for calculation

    Returns:
        next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
