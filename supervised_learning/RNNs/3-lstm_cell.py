#!/usr/bin/env python3
"""
Module defining the LSTMCell class
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit
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
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized as zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev (np.ndarray): shape (m, h) previous hidden state
            c_prev (np.ndarray): shape (m, h) previous cell state
            x_t (np.ndarray): shape (m, i) input data

        Returns:
            h_next (np.ndarray): next hidden state
            c_next (np.ndarray): next cell state
            y (np.ndarray): output of the cell
        """
        # Concatenate prev hidden state first, then input
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget Gate
        f_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wf) + self.bf)))

        # Update Gate
        u_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wu) + self.bu)))

        # Intermediate Cell State (Candidate)
        c_tilde = np.tanh(np.matmul(concat_input, self.Wc) + self.bc)

        # Next Cell State
        c_next = f_t * c_prev + u_t * c_tilde

        # Output Gate
        o_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wo) + self.bo)))

        # Next Hidden State
        h_next = o_t * np.tanh(c_next)

        # Output y with softmax activation
        y_logits = np.matmul(h_next, self.Wy) + self.by
        exp_logits = np.exp(y_logits - np.max(y_logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, c_next, y
