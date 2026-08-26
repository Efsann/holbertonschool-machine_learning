#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b: numpy.ndarray of shape (1, 1, 1, c_new)
    padding: string 'same' or 'valid'
    stride: tuple (sh, sw)

    Returns:
    tuple of (dA_prev, dW, db)
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                a_slice = A_prev_pad[:, vert_start:vert_end,
                                     horiz_start:horiz_end, :]

                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] \
                    += W[:, :, :, k] * dZ[:, i:i+1, j:j+1, k:k+1]

                dW[:, :, :, k] += np.sum(
                    a_slice * dZ[:, i:i+1, j:j+1, k:k+1],
                    axis=0
                )

    if padding == 'same' and (ph > 0 or pw > 0):
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
