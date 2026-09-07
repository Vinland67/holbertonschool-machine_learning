#!/usr/bin/env python3
"""
Module defining the sarsa_lambtha function
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines next action using epsilon-greedy policy

    Args:
        Q: numpy.ndarray containing Q-table
        state: current state
        epsilon: threshold for exploration

    Returns:
        next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(lambda) algorithm on the environment

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s, a) containing Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: updated Q table
    """
    initial_epsilon = epsilon
    E = np.zeros_like(Q)

    for episode in range(episodes):
        state, _ = env.reset()
        E.fill(0)
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            E[state, action] += 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q
