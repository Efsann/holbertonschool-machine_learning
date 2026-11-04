#!/usr/bin/env python3
"""
Optimize k Module
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: minimum number of clusters to check for (inclusive)
        kmax: maximum number of clusters to check for (inclusive)
        iterations: maximum number of iterations for K-means

    returns:
        results, d_vars, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    base_var = variances[0]
    d_vars = [base_var - v for v in variances]

    return results, d_vars
