#!/usr/bin/env python3
"""
Module to save and load an entire Keras model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    network: the model to save
    filename: path of the file the model should be saved to

    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model

    filename: path of the file the model should be loaded from

    Returns: the loaded model
    """
    return K.models.load_model(filename)
