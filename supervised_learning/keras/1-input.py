#!/usr/bin/env python3
"""
Module containing function to build a model with Keras input
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras layers.

    Parameters:
    nx: number of input features to the network
    layers: list containing the number of nodes in each layer of the network
    activations: list containing the activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Returns:
    The keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1 and keep_prob is not None:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
