#!/usr/bin/env python3
"""Module to calculate the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n).

    Returns:
        The string representing the definiteness category,
        or None if matrix is invalid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Matrisin etibarlılığını (valid matrix) yoxlayırıq:
    # 2D olmalıdır, kvadrat olmalıdır və boş olmamalıdır
    if (len(matrix.shape) != 2 or
            matrix.shape[0] != matrix.shape[1] or
            matrix.shape[0] == 0):
        return None

    # Matris mütləq simmetrik olmalıdır
    if not np.array_equal(matrix, matrix.T):
        return None

    # Məxsusi qiymətləri (eigenvalues) tapırıq
    w = np.linalg.eigvals(matrix)

    # Qruplara ayırırıq
    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    if np.any(w > 0) and np.any(w < 0):
        return "Indefinite"

    return None
