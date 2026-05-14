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
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Kovariasiya matrisinin (cov) (d, d) ölçüsündə hesablanması
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF at point x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # PDF düsturunun hissələri:
        # 1. Determinant və Tərs matris
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # 2. Normalizasiya sabiti (1 / sqrt((2*pi)^d * det))
        denominator = np.sqrt(((2 * np.pi) ** d) * det)

        # 3. Eksponent hissəsi: -0.5 * (x - mu).T * inv * (x - mu)
        diff = x - self.mean
        exponent_val = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)

        # Nəticə
        pdf_value = (1 / denominator) * np.exp(exponent_val)

        # pdf_value (1, 1) ölçülü massiv olduğu üçün skalyar qaytarırıq
        return pdf_value[0][0]
