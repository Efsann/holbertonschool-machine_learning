#!/usr/bin/env python3
"""
Module to perform pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'

    Returns:
    numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int((h - kh) / sh) + 1
    out_w = int((w - kw) / sw) + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = images[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            if mode == 'max':
                output[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return output
