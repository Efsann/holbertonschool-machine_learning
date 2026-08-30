#!/usr/bin/env python3
"""
Module defining the RNNCell class
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor for RNNCell

        Parameters:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        # Wh is for concatenated hidden state and input data: shape (h + i, h)
        self.Wh = np.random.normal(size=(h + i, h))
        # Wy is for output: shape (h, o)
        self.Wy = np.random.normal(size=(h, o))
        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev (np.ndarray): shape (m, h) previous hidden state
            x_t (np.ndarray): shape (m, i) data input for the cell

        Returns:
            h_next (np.ndarray): next hidden state
            y (np.ndarray): output of the cell
        """
        # Concatenate h_prev and x_t along columns (axis 1)
        # Shape: (m, h + i)
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state with tanh activation
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        # Compute output logits
        y_logits = np.dot(h_next, self.Wy) + self.by

        # Compute softmax activation for y
        exp_y = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
