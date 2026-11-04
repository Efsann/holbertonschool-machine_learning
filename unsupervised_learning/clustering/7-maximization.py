#!/usr/bin/env python3
"""
Maximization Step for GMM
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    returns:
        pi, m, S, or None, None, None on failure
            pi: numpy.ndarray of shape (k,) containing updated priors
            m: numpy.ndarray of shape (k, d) containing updated means
            S: numpy.ndarray of shape (k, d, d) containing updated covariances
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    N_k = np.sum(g, axis=1)

    pi = N_k / n

    m = np.matmul(g, X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))

    for j in range(k):
        X_m = X - m[j]
        S[j] = np.matmul(g[j] * X_m.T, X_m) / N_k[j]

    return pi, m, S
