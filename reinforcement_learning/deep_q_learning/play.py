#!/usr/bin/env python3
"""
Script to evaluate/display a game of Atari Breakout using trained weights
"""
import gymnasium as gym
from gymnasium import Wrapper
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


class GymToKerasRLWrapper(Wrapper):
    """
    Wrapper to adapt Gymnasium interface for keras-rl2 compatibility
    """
    def reset(self, **kwargs):
        """
        Resets the environment and returns only the observation tensor
        """
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """
        Performs an action step and adapts returns (obs, reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


def build_model(height, width, channels, actions):
    """
    Builds a convolutional neural network for processing Atari frames
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(1, height, width, channels)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def main():
    """
    Loads saved weights and plays Breakout using GreedyQPolicy
    """
    raw_env = gym.make('Atari/Breakout-v4', render_mode='human')
    env = GymToKerasRLWrapper(raw_env)

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    memory = SequentialMemory(limit=1000000, window_length=1)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        policy=policy
    )

    dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == '__main__':
    main()
