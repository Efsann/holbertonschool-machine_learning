#!/usr/bin/env python3
"""
Module to update weights of a neural network with Dropout regularization
using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent in place.

    Parameters:
    Y: one-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights: dictionary of weights and biases of the neural network
    cache: dictionary of outputs and dropout masks of each layer
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_curr = weights['W' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(W_curr.T, dZ)
            D_prev = cache['D' + str(i - 1)]
            dA = (dA * D_prev) / keep_prob
            dZ = dA * (1 - np.square(A_prev))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
