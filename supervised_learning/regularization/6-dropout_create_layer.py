#!/usr/bin/env python3
"""
Module to create a layer with Dropout in TensorFlow
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout in TensorFlow.

    Parameters:
    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function for the new layer
    keep_prob: probability that a node will be kept
    training: boolean indicating whether the model is in training mode

    Returns:
    The output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    output = layer(prev)

    if training:
        dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
        output = dropout(output, training=training)

    return output
