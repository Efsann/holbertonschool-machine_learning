#!/usr/bin/env python3
"""
Module to randomly adjust the contrast of an image
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    Parameters:
    image: 3D tf.Tensor representing the input image to adjust the contrast
    lower: float representing the lower bound of random contrast factor range
    upper: float representing the upper bound of random contrast factor range

    Returns:
    The contrast-adjusted image tensor
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
