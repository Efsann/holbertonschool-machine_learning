#!/usr/bin/env python3
"""
Module defining the BayesianOptimization class with acquisition functionality
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that performs Bayesian optimization on a noiseless 1D GP
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor for BayesianOptimization

        Parameters:
        - f: black-box function to be optimized
        - X_init: numpy.ndarray of shape (t, 1) representing inputs
        - Y_init: numpy.ndarray of shape (t, 1) representing outputs
        - bounds: tuple of (min, max) representing space bounds
        - ac_samples: number of acquisition samples
        - l: length parameter for the kernel
        - sigma_f: standard deviation for output
        - xsi: exploration-exploitation factor
        - minimize: bool determining minimization vs maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
        - X_next: numpy.ndarray of shape (1,) representing the next best sample
        - EI: numpy.ndarray of shape (ac_samples,) containing expected
              improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_opt = np.min(self.gp.Y)
            imp = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            imp = mu - y_opt - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei
