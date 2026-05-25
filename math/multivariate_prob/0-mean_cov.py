#!/usr/bin/env python3
"""Module to calculate the mean and covariance of a dataset."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
           n is the number of data points
           d is the number of dimensions in each data point

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # 1. Ortalamanın hesablanması (axis=0 sütunlar üzrə hesablayır)
    # keepdims=True edirik ki, nəticə (1, d) ölçüsündə olsun
    mean = np.mean(X, axis=0, keepdims=True)

    # 2. Kovariasiya matrisinin hesablanması
    # X_centered = X - mean
    X_centered = X - mean

    # cov = (X_centered.T * X_centered) / (n - 1)
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
