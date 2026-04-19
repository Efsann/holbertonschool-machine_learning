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
    # Siyahı olub-olmadığını və boş olmadığını yoxlayırıq
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    
    # Bütün elementlərin rəqəm (int və ya float) olub-olmadığını yoxlayırıq
    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    # Törəməni hesablayırıq (hər elementi öz indeksinə vururuq)
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # Əgər törəmə siyahısı boşdursa (yəni əsas siyahıda ancaq sabit rəqəm var idisə)
    if len(derivative) == 0:
        return [0]

    return derivative
