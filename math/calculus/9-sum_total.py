#!/usr/bin/env python3
"""
Module to calculate the sum of squares without loops.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n.
    """
    # N-in etibarlı rəqəm (tam və ya kəsr kimsiz) olub-olmadığını yoxlayırıq
    if type(n) not in (int, float) or int(n) != n or n < 1:
        return None
    
    n = int(n)
    
    # Riyazi düsturla hesablayıb nəticəni tam rəqəm (integer) kimi qaytarırıq
    return (n * (n + 1) * (2 * n + 1)) // 6

