#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    images: numpy.ndarray with shape (m, h, w, c)
    kernels: numpy.ndarray with shape (kh, kw, c, nc)
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw)

    Returns:
    numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = int((h + 2 * ph - kh) / sh) + 1
    out_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                image_slice = padded[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                ]
                output[:, i, j, k] = np.sum(
                    image_slice * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
