#!/usr/bin/env python3
"""
Module defining a noiseless 1D Gaussian Process with prediction
"""
import numpy as np


class GaussianProcess:
    """
    Class that represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor for GaussianProcess

        Parameters:
        - X_init: numpy.ndarray of shape (t, 1) representing inputs
        - Y_init: numpy.ndarray of shape (t, 1) representing outputs
        - l: length parameter for the kernel
        - sigma_f: standard deviation given to the output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using Radial Basis Function (RBF)

        Parameters:
        - X1: numpy.ndarray of shape (m, 1)
        - X2: numpy.ndarray of shape (n, 1)

        Returns:
        - Covariance kernel matrix as numpy.ndarray of shape (m, n)
        """
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1)
            + np.sum(X2**2, 1)
            - 2 * np.dot(X1, X2.T)
        )
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a GP

        Parameters:
        - X_s: numpy.ndarray of shape (s, 1) containing all sample points

        Returns:
        - mu: numpy.ndarray of shape (s,) containing the mean for each point
        - sigma: numpy.ndarray of shape (s,) containing the variance
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.matmul(K_s.T, np.matmul(K_inv, self.Y)).reshape(-1)
        sigma = np.diag(K_ss - np.matmul(K_s.T, np.matmul(K_inv, K_s)))

        return mu, sigma
