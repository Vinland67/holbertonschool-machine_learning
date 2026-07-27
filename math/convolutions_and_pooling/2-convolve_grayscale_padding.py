#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with
custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    images: numpy.ndarray of shape (m, h, w) containing multiple
        grayscale images
    kernel: numpy.ndarray of shape (kh, kw) containing the kernel
        for the convolution
    padding: tuple of (ph, pw)
        ph: padding for the height of the image
        pw: padding for the width of the image

    The image is padded with 0's.

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(images,
                     ((0, 0), (ph, ph), (pw, pw)),
                     mode='constant')

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel,
                                         axis=(1, 2))

    return convolved
