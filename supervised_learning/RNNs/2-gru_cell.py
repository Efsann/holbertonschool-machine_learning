#!/usr/bin/env python3
"""
Module defining the GRUCell class
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell
    """
    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

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
        concat_input = np.concatenate((x_t, h_prev), axis=1)

        # Update gate: z_t = sigmoid(concat_input @ Wz + bz)
        z_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wz) + self.bz)))

        # Reset gate: r_t = sigmoid(concat_input @ Wr + br)
        r_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wr) + self.br)))

        # Candidate/intermediate hidden state
        concat_reset = np.concatenate((x_t, r_t * h_prev), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state: h_next = (1 - z_t) * h_prev + z_t * h_tilde
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output with softmax activation
        y_logits = np.matmul(h_next, self.Wy) + self.by
        exp_logits = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
