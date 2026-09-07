#!/usr/bin/env python3
"""
Module defining the load_frozen_lake function
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium

    Args:
        desc: list of lists containing a custom description of the map, or None
        map_name: string containing the pre-made map to load, or None
        is_slippery: boolean to determine if the ice is slippery

    Returns:
        env: the gymnasium environment
    """
    if desc is None and map_name is None:
        env = gym.make(
            'FrozenLake-v1',
            map_name='8x8',
            is_slippery=is_slippery
        )
    else:
        env = gym.make(
            'FrozenLake-v1',
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery
        )

    return env
