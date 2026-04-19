#!/usr/bin/env python3
"""
Module to calculate the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): Coefficients of the polynomial.
        C (int): Integration constant.

    Returns:
        list: Coefficients of the integral, or None if invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int) and not isinstance(C, float):
        return None
    # C tam ədədə bərabərdirsə integer olmalıdır
    if isinstance(C, float) and C.is_integer():
        C = int(C)
    elif isinstance(C, float):
        return None

    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    # İnteqralın hesablanması
    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        # Əgər tam ədəddirsə .0-ı atırıq
        if val == int(val):
            integral.append(int(val))
        else:
            integral.append(val)

    # Siyahının sonundakı artıq sıfırları silirik (leading zeros)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
