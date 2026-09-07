#!/usr/bin/env python3
"""
Module defining the monte_carlo function
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next action
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
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        episode_data = np.array(episode_data, dtype=object)
        G = 0

        for t in range(len(episode_data) - 1, -1, -1):
            s_t, r_t = episode_data[t]
            G = gamma * G + r_t

            if s_t not in episode_data[:t, 0]:
                V[s_t] = V[s_t] + alpha * (G - V[s_t])

    return V
