#!/usr/bin/env python3
"""
PCA module for dimensionality reduction
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    parameters:
        X: numpy.ndarray of shape (n, d) where all dimensions have zero mean
        var: fraction of the variance that PCA should maintain

    returns:
        W: weights matrix of shape (d, nd)
    """
    u, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)

    idx = np.argwhere(cum_var >= var)[0][0]
    nd = idx + 1

    W = vh[:nd].T

    return W
