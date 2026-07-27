#!/usr/bin/env python3
"""
Module to rotate an image 90 degrees counter-clockwise
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise

    image: 3D tf.Tensor containing the image to rotate

    Returns: the rotated image
    """
    return tf.image.rot90(image, k=1)
