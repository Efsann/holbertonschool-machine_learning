  GNU nano 6.2                                                                                                    6-momentum.py *                                                                                                           
#!/usr/bin/env python3
"""
Module to set up Momentum optimization in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TF.

    Parameters:
    alpha: learning rate
    beta1: momentum weight

    Returns:
    The optimizer
    """

