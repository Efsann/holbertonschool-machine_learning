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
        # Concatenated shape is (i + h, h) -> x_t (i) and h_prev (h) concatenated
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev (np.ndarray): shape (m, h) containing previous hidden state
            x_t (np.ndarray): shape (m, i) containing input data

        Returns:
            h_next (np.ndarray): next hidden state
            y (np.ndarray): output of the cell
        """
        # Concatenate x_t and h_prev along axis 1 -> shape (m, i + h)
        concat_input = np.concatenate((x_t, h_prev), axis=1)

        # Calculate next hidden state using tanh
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate output logits
        y_logits = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation function
        exp_logits = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
