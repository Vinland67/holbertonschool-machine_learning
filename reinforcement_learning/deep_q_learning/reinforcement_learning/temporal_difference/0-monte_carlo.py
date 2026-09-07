#!/usr/bin/env python3
"""
Monte Carlo Algorithm for Reinforcement Learning
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm on an environment.

    Parameters:
    - env: the environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: function that takes a state and returns the next action
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount rate

    Returns:
    - V: updated value estimate
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        states = [x[0] for x in episode_data]
        rewards = [x[1] for x in episode_data]

        G = 0
        for t in range(len(episode_data) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]

            # First-visit Monte Carlo check
            if s not in states[:t]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
