#!/usr/bin/env python3
"""
GMM with scikit-learn
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using scikit-learn

    parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters

    returns:
        pi, m, S, clss, bic
            pi: numpy.ndarray of shape (k,) containing cluster priors
            m: numpy.ndarray of shape (k, d) containing centroid means
            S: numpy.ndarray of shape (k, d, d) containing covariance matrices
            clss: numpy.ndarray of shape (n,) containing cluster indices
            bic: BIC value for the model
    """
    g = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)

    return pi, m, S, clss, bic
