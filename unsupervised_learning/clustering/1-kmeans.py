#!/usr/bin/env python3
"""
K-means Clustering Module
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: maximum number of iterations

    returns:
        C, clss, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for i in range(iterations):
        C_old = np.copy(C)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = np.random.uniform(low=low, high=high)
            else:
                C[j] = np.mean(points, axis=0)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_old):
            break

    return C, clss

