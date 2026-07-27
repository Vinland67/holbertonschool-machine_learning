#!/usr/bin/env python3
"""
Module to perform back propagation over a pooling layer
of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the output
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer
    kernel_shape: tuple of (kh, kw)
        containing the size of the kernel for the pooling
    stride: tuple of (sh, sw)
        containing the strides for the pooling
    mode: string containing either 'max' or 'avg'

    Returns: partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=float)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_prev_slice = a_prev[
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            f
                        ]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            f
                        ] += mask * dA[i, h, w, f]

                    elif mode == 'avg':
                        da = dA[i, h, w, f]
                        average_grad = da / (kh * kw)
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            f
                        ] += np.ones((kh, kw)) * average_grad

    return dA_prev
