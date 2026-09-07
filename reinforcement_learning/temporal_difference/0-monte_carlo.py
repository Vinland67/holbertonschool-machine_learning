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
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Əgər agent deşiyə düşübsə (reward 0-dır amma oyun bitib)
            if terminated and reward == 0:
                reward = V[next_state]  # Son vəziyyətin öz dəyərini (-1) götürürük
            # Əgər agent hədəfə çatıbsa
            elif terminated and reward == 1:
                reward = V[next_state]  # Hədəfin dəyərini (1) götürürük

            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        # Hər bir epizod bitdikdən sonra Every-Visit yeniləməsi edirik
        for s_t, r_t in reversed(episode_data):
            G = gamma * G + r_t
            V[s_t] = V[s_t] + alpha * (G - V[s_t])

    return V
