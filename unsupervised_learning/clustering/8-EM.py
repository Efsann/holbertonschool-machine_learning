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
        pi, m, S, g, l, or None, None, None, None, None on failure
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

    l_prev = 0
    i = 0

    for i in range(iterations):
        g, l = expectation(X, pi, m, S)  # noqa: E741
        if g is None or l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if i > 0 and abs(l - l_prev) <= tol:
            break

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = l
    else:
        g, l = expectation(X, pi, m, S)  # noqa: E741
        if g is None or l is None:
            return None, None, None, None, None

    if verbose and (i % 10 != 0 or i == iterations - 1):
        print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

    return pi, m, S, g, l
