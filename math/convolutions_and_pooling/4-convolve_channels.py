#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels

    images: numpy.ndarray of shape (m, h, w, c) containing multiple
        images
    kernel: numpy.ndarray of shape (kh, kw, c) containing the kernel
        for the convolution
    padding: either a tuple of (ph, pw), 'same', or 'valid'
    stride: tuple of (sh, sw)

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row = i * sh
            col = j * sw
            image_slice = padded[:, row:row + kh, col:col + kw, :]
            convolved[:, i, j] = np.sum(
                image_slice * kernel, axis=(1, 2, 3))

    return convolved
