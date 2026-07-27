#!/usr/bin/env python3
"""
Module to perform pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images: numpy.ndarray of shape (m, h, w, c) containing multiple
        images
    kernel_shape: tuple of (kh, kw) containing the kernel shape for
        the pooling
    stride: tuple of (sh, sw)
    mode: type of pooling ('max' or 'avg')

    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            row = i * sh
            col = j * sw
            image_slice = images[:, row:row + kh, col:col + kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
