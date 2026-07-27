#!/usr/bin/env python3
"""
Module to randomly change the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    image: 3D tf.Tensor containing the image to change
    max_delta: maximum amount the image should be brightened (or darkened)

    Returns: the altered image
    """
    return tf.image.random_brightness(image, max_delta)
