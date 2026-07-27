#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer of
a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a
    neural network

    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the unactivated
        output of the convolutional layer
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
        biases applied to the convolution
    padding: string, either "same" or "valid"
    stride: tuple of (sh, sw) containing the strides for the
        convolution

    Returns: the partial derivatives with respect to the previous
        layer (dA_prev), the kernels (dW), and the biases (db),
        respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')

    dA_prev_padded = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                row = i * sh
                col = j * sw
                image_slice = padded[:, row:row + kh, col:col + kw, :]

                dW[:, :, :, k] += np.sum(
                    image_slice * dZ[:, i, j, k].reshape(-1, 1, 1, 1),
                    axis=0)

                dA_prev_padded[:, row:row + kh, col:col + kw, :] += (
                    W[:, :, :, k] *
                    dZ[:, i, j, k].reshape(-1, 1, 1, 1))

    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
