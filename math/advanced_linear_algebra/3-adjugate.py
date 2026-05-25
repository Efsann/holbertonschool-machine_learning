#!/usr/bin/env python3
"""Module to calculate the adjugate matrix of a matrix."""


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


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.

    Args:
        matrix: list of lists whose adjugate matrix should be calculated.

    Returns:
        The adjugate matrix.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    # 1. Kofaktor matrisini hesablamaq
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            minor_val = determinant(sub)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_val)
        cofactor_matrix.append(cofactor_row)

    # 2. Kofaktor matrisini transponirə edib Adjugate almaq
    adjugate_matrix = []
    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
