#!/usr/bin/env python3
"""
Module to train a Keras model using early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping.

    network: the model to train
    data: numpy.ndarray containing the input data
    labels: one-hot numpy.ndarray containing the labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: number of passes through data
    validation_data: data to validate the model with, if not None
    early_stopping: boolean indicating whether to use early stopping
    patience: patience used for early stopping
    verbose: boolean that determines if output should be printed
    shuffle: boolean that determines whether to shuffle the batches
    Returns: the History object generated after training the model
    """
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
