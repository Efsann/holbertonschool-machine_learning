#!/usr/bin/env python3
"""
Bayesian Information Criterion Module
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: minimum number of clusters to check for (inclusive)
        kmax: maximum number of clusters to check for (inclusive)
        iterations: maximum number of iterations for EM
        tol: tolerance for EM
        verbose: boolean determining if EM prints information

    returns:
        best_k, best_result, log_likes, bics, or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    log_likes = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None:
            return None, None, None, None

        p = k * (1 + d + (d * (d + 1)) // 2) - 1
        bic = p * np.log(n) - 2 * log_like

        log_likes.append(log_like)
        bics.append(bic)
        results.append((pi, m, S))

    log_likes = np.array(log_likes)
    bics = np.array(bics)

    best_idx = np.argmin(bics)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likes, bics
