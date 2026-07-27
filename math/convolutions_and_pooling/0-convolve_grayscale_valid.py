#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    images: numpy.ndarray of shape (m, h, w) containing multiple
        grayscale images
    kernel: numpy.ndarray of shape (kh, kw) containing the kernel
        for the convolution

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(
                image_slice * kernel, axis=(1, 2))

    return convolved
