#!/usr/bin/env python3
"""
Module to save and load a model's weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights

    network: the model whose weights should be saved
    filename: path of the file the weights should be saved to
    save_format: format in which the weights should be saved

    Returns: None
    """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights

    network: the model to which the weights should be loaded
    filename: path of the file the weights should be loaded from

    Returns: None
    """
    network.load_weights(filename)
    return None
