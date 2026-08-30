#!/usr/bin/env python3
"""
Module defining the RNNCell class for a simple RNN
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        # Concatenated shape (h + i, h) -> h_prev (h) and x_t (i) concatenated
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev (np.ndarray): shape (m, h) previous hidden state
            x_t (np.ndarray): shape (m, i) input data

        Returns:
            h_next (np.ndarray): next hidden state
            y (np.ndarray): output of the cell
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        y_logits = np.matmul(h_next, self.Wy) + self.by

        exp_logits = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
