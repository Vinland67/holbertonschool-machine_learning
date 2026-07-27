#!/usr/bin/env python3
"""
Module to randomly adjust the contrast of an image
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    image: 3D tf.Tensor representing the input image
    lower: float representing the lower bound of contrast factor range
    upper: float representing the upper bound of contrast factor range

    Returns: contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
