#!/usr/bin/env python3
"""
Module to update weights and biases using GD with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization in place.

    Parameters:
    Y: one-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights: dictionary of weights and biases of the neural network
    cache: dictionary of the outputs of each layer of the neural network
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_curr = weights['W' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W_curr
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.matmul(W_curr.T, dZ) * (1 - np.square(A_prev))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
