#!/usr/bin/env python3
"""
Module to save and load a model's configuration in JSON format
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

    network: the model whose configuration should be saved
    filename: path of the file the configuration should be saved to

    Returns: None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration

    filename: path of the file containing the model's configuration
        in JSON format

    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
