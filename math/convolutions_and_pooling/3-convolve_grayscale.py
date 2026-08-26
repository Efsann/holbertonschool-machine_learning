#!/usr/bin/env python3
"""
Module to perform a strided convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw)

    Returns:
    numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
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
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    out_h = int((h + 2 * ph - kh) / sh) + 1
    out_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
