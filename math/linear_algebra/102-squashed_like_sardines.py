#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    def get_shape(matrix):
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            if len(matrix) == 0:
                break
            matrix = matrix[0]
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Ölçülərin sayını yoxla
    if len(shape1) != len(shape2):
        return None

    # Seçilmiş oxdan başqa digər oxların bərabərliyini yoxla
    for i in range(len(shape1)):
        if i != axis:
            if shape1[i] != shape2[i]:
                return None

    def cat_recursive(m1, m2, current_axis):
        if current_axis == axis:
            return m1 + m2
        return [cat_recursive(m1[i], m2[i], current_axis + 1)
                for i in range(len(m1))]

    try:
        return cat_recursive(mat1, mat2, 0)
    except Exception:
        return None
