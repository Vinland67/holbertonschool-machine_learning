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
    img_float = tf.cast(image, dtype=tf.float32)
    reshaped_img = tf.reshape(img_float, (-1, 3))

    # Mean centering
    mean = tf.reduce_mean(reshaped_img, axis=0)
    centered = reshaped_img - mean

    # Covariance matrix calculation
    cov = tf.matmul(centered, centered, transpose_a=True)
    cov = cov / tf.cast(tf.shape(reshaped_img)[0] - 1, tf.float32)

    # Singular Value Decomposition (SVD) or Eigen decomposition
    e_val, e_vec = tf.linalg.eigh(cov)

    # Compute perturbation offset: e_vec * (alphas * e_val)
    alphas = tf.cast(alphas, dtype=tf.float32)
    delta = tf.matmul(e_vec, tf.reshape(alphas * e_val, (3, 1)))
    delta = tf.reshape(delta, (1, 1, 3))

    # Apply perturbation, clip to [0, 255], convert to uint8
    augmented = tf.clip_by_value(img_float + delta, 0, 255)
    return tf.cast(augmented, dtype=tf.uint8)
