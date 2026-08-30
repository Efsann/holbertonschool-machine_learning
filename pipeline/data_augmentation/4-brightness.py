#!/usr/bin/env python3
"""
Module to randomly change the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Parameters:
    image: 3D tf.Tensor containing the image to change
    max_delta: maximum amount the image should be brightened (or darkened)

    Returns:
    The altered image tensor
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
