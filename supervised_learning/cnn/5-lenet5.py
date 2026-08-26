#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 network using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.

    Parameters:
    X: K.Input of shape (m, 28, 28, 1)

    Returns:
    K.Model compiled with Adam optimizer and accuracy metrics
    """
    init = K.initializers.HeNormal(seed=0)

    # 1. Conv layer: 6 kernels, 5x5, padding='same', relu
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # 2. Max pooling: 2x2, stride=(2, 2)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # 3. Conv layer: 16 kernels, 5x5, padding='valid', relu
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # 4. Max pooling: 2x2, stride=(2, 2)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten before fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # 5. Fully connected: 120 nodes, relu
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # 6. Fully connected: 84 nodes, relu
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # 7. Fully connected softmax output layer: 10 nodes
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
