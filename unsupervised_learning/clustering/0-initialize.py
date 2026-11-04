#!/usr/bin/env python3
"""
K-means Initialization Module
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    centroids = np.random.uniform(low=low, high=high, size=(k, X.shape[1]))

    return centroids
