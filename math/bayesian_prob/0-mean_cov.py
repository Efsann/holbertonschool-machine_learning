#!/usr/bin/env python3
"""Module for calculating mean and covariance matrix."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set

    Returns:
        mean: numpy.ndarray of shape (1, d)
        cov: numpy.ndarray of shape (d, d)
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n = X.shape[0]

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean hesablanması
    mean = np.mean(X, axis=0, keepdims=True)

    # Mean çıxılır
    X_centered = X - mean

    # Covariance matrix hesablanması
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
