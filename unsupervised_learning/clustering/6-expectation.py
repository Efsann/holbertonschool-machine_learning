#!/usr/bin/env python3
"""
Expectation Step for GMM
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    returns:
        g, l, or None, None on failure
            g: numpy.ndarray of shape (k, n) containing posterior probabilities
            l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1) or np.any(pi < 0):
        return None, None

    g = np.zeros((k, n))

    for j in range(k):
        P = pdf(X, m[j], S[j])
        if P is None:
            return None, None
        g[j] = pi[j] * P

    marginal = np.sum(g, axis=0)
    l = np.sum(np.log(marginal))
    g /= marginal

    return g, l
