#!/usr/bin/env python3
"""
Expectation Maximization for GMM Module
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: maximum number of iterations for the algorithm
        tol: tolerance of the log likelihood for early stopping
        verbose: boolean determining if info should be printed

    returns:
        pi, m, S, g, log_like, or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_like = expectation(X, pi, m, S)
    if g is None or log_like is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(log_like))

    for i in range(1, iterations + 1):
        l_prev = log_like

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_like = expectation(X, pi, m, S)
        if g is None or log_like is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or abs(log_like - l_prev) <= tol
                        or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, log_like))

        if abs(log_like - l_prev) <= tol:
            break

    return pi, m, S, g, log_like
