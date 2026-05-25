#!/usr/bin/env python3
"""Module to calculate a correlation matrix."""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
           d is the number of dimensions

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Dispersiyalar matrisin əsas diaqonalındadır
    variance = np.diag(C)

    # Standart meylləşmə dispersiyanın kvadrat köküdür
    std_dev = np.sqrt(variance)

    # Məxrəc (sigma_X * sigma_Y) üçün outer product
    std_dev_matrix = np.outer(std_dev, std_dev)

    # Korrelyasiya = Kovariasiya / (sigma_X * sigma_Y)
    correlation_matrix = C / std_dev_matrix

    return correlation_matrix
