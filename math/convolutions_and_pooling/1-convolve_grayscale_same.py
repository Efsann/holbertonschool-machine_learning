#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)

    Returns:
    numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(kh / 2) if kh % 2 != 0 else int((kh - 1) / 2)
    pw = int(kw / 2) if kw % 2 != 0 else int((kw - 1) / 2)

    # Dəqiq same konvolusiya padding hesablanması
    ph_top = int(np.ceil((kh - 1) / 2))
    ph_bottom = int(np.floor((kh - 1) / 2))
    pw_left = int(np.ceil((kw - 1) / 2))
    pw_right = int(np.floor((kw - 1) / 2))

    padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
