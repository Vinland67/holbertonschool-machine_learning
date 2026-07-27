#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    images: numpy.ndarray of shape (m, h, w, c) containing multiple
        images
    kernels: numpy.ndarray of shape (kh, kw, c, nc) containing the
        kernels for the convolution
    padding: either a tuple of (ph, pw), 'same', or 'valid'
    stride: tuple of (sh, sw)

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
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

    convolved = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                row = i * sh
                col = j * sw
                image_slice = padded[:, row:row + kh, col:col + kw, :]
                convolved[:, i, j, k] = np.sum(
                    image_slice * kernels[:, :, :, k], axis=(1, 2, 3))

    return convolved
