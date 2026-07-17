#!/usr/bin/env python3
"""
Module to train a Keras model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    network: the model to train
    data: numpy.ndarray containing the input data
    labels: one-hot numpy.ndarray containing the labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data
    verbose: boolean that determines if output should be printed
    shuffle: boolean that determines whether to shuffle the batches
    Returns: the History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
