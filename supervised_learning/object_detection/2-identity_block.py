#!/usr/bin/env python3
"""
Module to build an identity block for ResNet using Keras
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)

    Parameters:
    - A_prev: output tensor from the previous layer
    - filters: tuple or list containing [F11, F3, F12]

    Returns:
    - The activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # First component of main path (1x1 conv)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path (3x3 conv)
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path (1x1 conv)
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut value to main path, then apply ReLU activation
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
