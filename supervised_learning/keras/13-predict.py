#!/usr/bin/env python3
"""
Module containing function to make a prediction using a Keras neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Parameters:
    network: The network model to make the prediction with.
    data: The input data to make the prediction with.
    verbose: Boolean determining if output should be printed during
             the prediction process.

    Returns:
    The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
