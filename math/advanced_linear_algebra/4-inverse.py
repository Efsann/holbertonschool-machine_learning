#!/usr/bin/env python3
"""Module to calculate the inverse of a matrix."""


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


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Args:
        matrix: list of lists whose inverse should be calculated.

    Returns:
        The inverse of the matrix, or None if the matrix is singular.
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

    # 1. Determinantı yoxlayırıq
    det = determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    # 2. Kofaktor matrisini tapırıq
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            minor_val = determinant(sub)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_val)
        cofactor_matrix.append(cofactor_row)

    # 3. Adjugate (Əlavə) matrisi tapırıq (Kofaktorun transponirə edilmişi)
    adjugate_matrix = []
    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    # 4. İnverse (Tərs) matrisi tapırıq (Adjugate / Determinant)
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adjugate_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
