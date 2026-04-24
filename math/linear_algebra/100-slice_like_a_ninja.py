#!/usr/bin/env python3
"""Slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Returns a new sliced numpy.ndarray"""
    # Bütün ölçülər üçün default dilim yaradırıq (:)
    slices = [slice(None)] * matrix.ndim

    # Verilmiş dictionary üzrə dilimləri yeniləyirik
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)

    return matrix[tuple(slices)]
