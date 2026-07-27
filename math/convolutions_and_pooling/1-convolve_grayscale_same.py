#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images: numpy.ndarray of shape (m, h, w) containing multiple
        grayscale images
    kernel: numpy.ndarray of shape (kh, kw) containing the kernel
        for the convolution

    If necessary, the image is padded with 0's.

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(images,
                     ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                     mode='constant')

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel,
                                         axis=(1, 2))

    return convolved
