#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies two 2D matrices and returns a new matrix"""
    if len(mat1[0]) != len(mat2):
        return None

    # Nəticə matrisini (m x p) sıfırlarla yaradırıq
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Matris vurma alqoritmi
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
