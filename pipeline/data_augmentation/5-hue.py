#!/usr/bin/env python3
"""
Module to change the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    image: 3D tf.Tensor containing the image to change
    delta: amount the hue should change

    Returns: the altered image
    """
    return tf.image.adjust_hue(image, delta)
