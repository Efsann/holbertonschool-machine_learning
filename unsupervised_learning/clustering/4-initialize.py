#!/usr/bin/env python3
"""
Initialize GMM Module
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters

    returns:
        pi, m, S, or None, None, None on failure
            pi: numpy.ndarray of shape (k,) containing cluster priors
            m: numpy.ndarray of shape (k, d) containing centroid means
            S: numpy.ndarray of shape (k, d, d) containing covariances
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    pi = np.full((k,), 1.0 / k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
