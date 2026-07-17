#!/usr/bin/env python3
"""
Module to save and load Keras models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network: The Keras model to be saved.
        filename: The path of the file where the model should be saved.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename: The path of the file from which the model should be loaded.

    Returns:
        The loaded Keras model.
    """
    return K.models.load_model(filename)
