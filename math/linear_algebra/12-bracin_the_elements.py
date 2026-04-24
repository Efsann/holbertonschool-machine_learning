#!/usr/bin/env python3
"""Performs element-wise operations using numpy"""


def np_elementwise(mat1, mat2):
    """Returns a tuple containing sum, diff, product, and quotient"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
