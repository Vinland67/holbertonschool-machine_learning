#!/usr/bin/env python3
"""
Module to flip an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally

    image: 3D tf.Tensor containing the image to flip

    Returns: the flipped image
    """
    return tf.image.flip_left_right(image)
