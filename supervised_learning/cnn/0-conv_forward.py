#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b: numpy.ndarray of shape (1, 1, 1, c_new)
    activation: activation function applied to the convolution
    padding: string 'same' or 'valid'
    stride: tuple (sh, sw)

    Returns:
    The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = int((h_prev + 2 * ph - kh) / sh) + 1
    out_w = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_new):
                slice_A = padded[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                ]
                Z[:, i, j, k] = np.sum(
                    slice_A * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]

    return activation(Z)
