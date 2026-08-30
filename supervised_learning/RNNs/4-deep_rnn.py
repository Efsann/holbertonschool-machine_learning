#!/usr/bin/env python3
"""
Module defining the deep_rnn function
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
        rnn_cells (list): list of RNNCell instances of length l
        X (np.ndarray): shape (t, m, i) input data
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0 (np.ndarray): shape (l, m, h) initial hidden state
            l: number of layers
            h: dimensionality of the hidden state

    Returns:
        H (np.ndarray): shape (t + 1, l, m, h) all hidden states
        Y (np.ndarray): shape (t, m, o) all outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # Initialize H with shape (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Determine output dimensionality from the last cell's output weights
    o = rnn_cells[-1].Wy.shape[1]
    Y = np.zeros((t, m, o))

    # Iterate through each time step
    for step in range(t):
        # Current input at time step t starts as X[step]
        x_t = X[step]

        # Iterate through each layer of the deep RNN
        for layer in range(l):
            # Previous hidden state for this layer
            h_prev = H[step, layer]

            # Forward pass through the current RNN cell
            h_next, y_next = rnn_cells[layer].forward(h_prev, x_t)

            # Store the updated hidden state for the next time step
            H[step + 1, layer] = h_next

            # Input for next layer is current layer's hidden state
            x_t = h_next

        # The output of the deep RNN at this step comes from the last layer
        Y[step] = y_next

    return H, Y
