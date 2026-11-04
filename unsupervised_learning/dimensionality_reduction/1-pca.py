#!/usr/bin/env python3
"""
PCA v2 module for dimensionality reduction
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce it to ndim dimensions

    parameters:
        X: numpy.ndarray of shape (n, d)
        ndim: new dimensionality of the transformed X

    returns:
        T: numpy.ndarray of shape (n, ndim) containing transformed X
    """
    X_mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_mean)
    W = vh[:ndim].T
    T = np.matmul(X_mean, W)

    return T
