#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer of a
neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a
    neural network

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    kernel_shape: tuple of (kh, kw) containing the size of the
        kernel for the pooling
    stride: tuple of (sh, sw) containing the strides for the pooling
    mode: string, either 'max' or 'avg'

    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            row = i * sh
            col = j * sw
            image_slice = A_prev[:, row:row + kh, col:col + kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
