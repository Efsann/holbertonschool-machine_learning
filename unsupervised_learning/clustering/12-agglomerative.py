#!/usr/bin/env python3
"""
Agglomerative Clustering Module
"""
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        Z, t=dist, criterion='distance'
    ) - 1

    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
