#!/usr/bin/env python3
"""
Module to calculate the sum of squares without loops.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.
    """
    if type(n) not in (int, float) or int(n) != n or n < 1:
        return None

    n = int(n)

    return (n * (n + 1) * (2 * n + 1)) // 6
