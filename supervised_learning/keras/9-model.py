#!/usr/bin/env python3
"""
Module containing functions to save and load Keras models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Parameters:
    network: The model to save.
    filename: The path of the file that the model should be saved to.

    Returns:
    None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Parameters:
    filename: The path of the file that the model should be loaded from.

    Returns:
    The loaded Keras model.
    """
    return K.models.load_model(filename)
