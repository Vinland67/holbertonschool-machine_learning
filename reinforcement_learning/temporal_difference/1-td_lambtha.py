#!/usr/bin/env python3
"""
Module defining the td_lambtha function
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm for value estimation

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next action
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    n_states = V.shape[0]

    for episode in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1

            V += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
