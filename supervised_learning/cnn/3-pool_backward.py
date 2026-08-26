#!/usr/bin/env python3
"""
Module to perform back propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
    dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: string 'max' or 'avg'

    Returns:
    dA_prev: partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        a_slice = a_prev[vert_start:vert_end,
                                         horiz_start:horiz_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += \
                            mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        avg_da = da / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += \
                            np.ones((kh, kw)) * avg_da

    return dA_prev
