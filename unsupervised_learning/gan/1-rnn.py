#!/usr/bin/env python3
"""
Module defining the rnn function for forward propagation
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Parameters:
        rnn_cell (RNNCell): instance of RNNCell used for forward propagation
        X (np.ndarray): input data of shape (t, m, i)
        h_0 (np.ndarray): initial hidden state of shape (m, h)

    Returns:
        H (np.ndarray): contains all hidden states, shape (t + 1, m, h)
        Y (np.ndarray): contains all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    o = rnn_cell.Wy.shape[1]
    Y = np.zeros((t, m, o))

    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y
        h_prev = h_next

    return H, Y
