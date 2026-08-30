#!/usr/bin/env python3
"""
Module defining the GRUCell class
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU)
    """
    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        # Weights initialized using a random normal distribution
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized as zeros
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
        # Concatenate prev hidden state first, then input
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate
        z_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wz) + self.bz)))

        # Reset Gate
        r_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wr) + self.br)))

        # Concatenate reset-gated hidden state with input
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)

        # Candidate Hidden State
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next Hidden State
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output y with softmax activation
        y_logits = np.matmul(h_next, self.Wy) + self.by
        exp_logits = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
