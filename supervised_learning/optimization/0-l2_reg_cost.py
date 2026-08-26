#!/usr/bin/env python3
"""
Module to calculate the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of weights and biases of the neural network
    L: number of layers in the neural network
    m: number of data points used

    Returns:
    The cost of the network accounting for L2 regularization
    """
    l2_sum = 0
    for key, value in weights.items():
        if key.startswith('W'):
            l2_sum += np.sum(np.square(value))

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
