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
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0 (np.ndarray): initial hidden state of shape (m, h)
            h: dimensionality of the hidden state

    Returns:
        H (np.ndarray): contains all hidden states, shape (t + 1, m, h)
        Y (np.ndarray): contains all outputs, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize H to store initial hidden state and all subsequent states
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Determine output dimensionality o by doing a temporary forward pass
    # or referencing the cell's Wy weight shape
    o = rnn_cell.Wy.shape[1]

    # Initialize Y to store outputs for all time steps
    Y = np.zeros((t, m, o))

    # Perform forward propagation through all time steps
    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y
        h_prev = h_next

    return H, Y
