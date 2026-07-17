#!/usr/bin/env python3
"""
Module containing function to test a Keras neural network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model.

    Parameters:
    network: The network model to test.
    data: The input data to test the model with.
    labels: The correct one-hot labels of data.
    verbose: Boolean determining if output should be printed during testing.

    Returns:
    The loss and accuracy of the model with the testing data, respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
