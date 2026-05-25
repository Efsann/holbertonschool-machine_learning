#!/usr/bin/env python3
"""Adds two matrices of any dimension element-wise"""


def add_matrices(mat1, mat2):
    """Adds two matrices of same shape and returns a new matrix"""
    # Ölçülərin uyğunluğunu yoxlayan köməkçi funksiya (Task 3-ə bənzər)
    def get_shape(matrix):
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            matrix = matrix[0]
        return shape

    if get_shape(mat1) != get_shape(mat2):
        return None

    # Əsas rekursiv toplama məntiqi
    def add_recursive(m1, m2):
        if isinstance(m1, (int, float)):
            return m1 + m2
        return [add_recursive(m1[i], m2[i]) for i in range(len(m1))]

    return add_recursive(mat1, mat2)
