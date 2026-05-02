#!/usr/bin/env python3
"""Module to calculate the minor matrix of a matrix."""


def determinant(matrix):
    """Helper function to calculate the determinant."""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    det = 0
    for c in range(n):
        sub = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.

    Args:
        matrix: list of lists whose minor matrix should be calculated.

    Returns:
        The minor matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
