#!/usr/bin/env python3
"""
Module for PCA color augmentation as described in the AlexNet paper
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper

    image: 3D tf.Tensor containing the image to change
    alphas: tuple of length 3 containing the amount that each channel
            should change

    Returns: the augmented image
    """
    # Flattens the image pixels to shape (N, 3) where N = height * width
    orig_dtype = image.dtype
    img_float = tf.cast(image, dtype=tf.float32)
    pixels = tf.reshape(img_float, (-1, 3))

    # Mean center the pixels
    mean = tf.reduce_mean(pixels, axis=0)
    centered_pixels = pixels - mean

    # Compute covariance matrix (3, 3)
    num_pixels = tf.cast(tf.shape(pixels)[0], tf.float32)
    cov = tf.matmul(centered_pixels, centered_pixels, transpose_a=True)
    cov = cov / (num_pixels - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    # Calculate delta offset: [p1, p2, p3] * ([alpha1, alpha2, alpha3] * [lambda1, lambda2, lambda3])
    alphas = tf.cast(alphas, dtype=tf.float32)
    delta = tf.matmul(
        eigenvectors,
        tf.reshape(alphas * eigenvalues, (3, 1))
    )

    # Reshape delta to (1, 1, 3) and add to image
    delta = tf.reshape(delta, (1, 1, 3))
    augmented_image = img_float + delta

    # Clip values to valid range [0, 255] and restore original dtype
    augmented_image = tf.clip_by_value(augmented_image, 0, 255)
    return tf.cast(augmented_image, dtype=orig_dtype)
