#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients representing a polynomial.

    Returns:
        list: A new list of coefficients representing the derivative.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for x in poly:
        if type(x) not in (int, float):
            return None

    derivative = [poly[i] * i for i in range(1, len(poly))]

    # Eger toreme siyahisi bosdursa
    if len(derivative) == 0:
        return [0]

    return derivative
