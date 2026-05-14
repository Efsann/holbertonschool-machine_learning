#!/usr/bin/env python3
"""Module that defines the MultiNormal class."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Class constructor.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                  n is the number of data points
                  d is the number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Ortalamanın (mean) (d, 1) ölçüsündə hesablanması
        # axis=1 sütunlar (n) üzrə ortalamanı tapır
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Kovariasiya matrisinin (cov) (d, d) ölçüsündə hesablanması
        # Məlumatın mərkəzləşdirilməsi
        data_centered = data - self.mean

        # cov = ( (X - mu) * (X - mu).T ) / (n - 1)
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)
