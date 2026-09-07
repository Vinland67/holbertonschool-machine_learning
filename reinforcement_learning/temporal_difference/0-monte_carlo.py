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
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            next_state, reward, done = step_res[0], step_res[1], step_res[2]

            episode_data.append((state, reward))

            if done:
                break

            state = next_state

        states = [x[0] for x in episode_data]
        rewards = [x[1] for x in episode_data]

        G = 0
        for t in range(len(episode_data) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]

            if s not in states[:t]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
