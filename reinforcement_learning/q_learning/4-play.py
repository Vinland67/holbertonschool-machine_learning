#!/usr/bin/env python3
"""
Module defining the play function
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using the Q-table

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total rewards for the episode
        rendered_outputs: list of rendered outputs representing board states
    """
    rendered_outputs = []
    state, _ = env.reset()

    rendered_outputs.append(env.render())

    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        rendered_outputs.append(env.render())

        if terminated or truncated:
            return reward, rendered_outputs

    return reward, rendered_outputs
