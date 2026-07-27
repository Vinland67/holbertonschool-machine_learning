#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer
of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
    neural network

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
        biases applied to the convolution
    activation: activation function applied to the convolution
    padding: string, either "same" or "valid"
    stride: tuple of (sh, sw) containing the strides for the
        convolution

    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')

    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_new):
                row = i * sh
                col = j * sw
                image_slice = padded[:, row:row + kh, col:col + kw, :]
                Z[:, i, j, k] = np.sum(
                    image_slice * W[:, :, :, k], axis=(1, 2, 3))

    Z = Z + b
    return activation(Z)
