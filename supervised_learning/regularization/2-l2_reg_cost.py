#!/usr/bin/env python3
"""
Module to calculate the cost of a Keras neural network with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization in Keras.

    Parameters:
    cost: tensor containing the cost of the network without L2 regularization
    model: Keras model that includes layers with L2 regularization

    Returns:
    A tensor containing the total cost accounting for L2 regularization
    """
    return cost + model.losses
