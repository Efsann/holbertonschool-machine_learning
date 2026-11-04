#!/usr/bin/env python3
"""
K-means with scikit-learn
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using scikit-learn

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    returns:
        C, clss
            C: numpy.ndarray of shape (k, d) containing centroid means
            clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = km.cluster_centers_
    clss = km.labels_

    return C, clss
