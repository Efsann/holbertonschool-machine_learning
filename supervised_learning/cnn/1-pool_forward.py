#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    kernel_shape: tuple of (kh, kw) containing the size of the kernel
    stride: tuple of (sh, sw) containing the strides
    mode: string 'max' or 'avg'

    Returns:
    The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int((h_prev - kh) / sh) + 1
    out_w = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            slice_A = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A
