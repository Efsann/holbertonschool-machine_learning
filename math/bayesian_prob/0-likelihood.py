#!/usr/bin/env python3
"""Module for computing Bayesian Probability Likelihood."""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: a 1D numpy.ndarray containing the various probabilities

    Returns:
        A 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        # Sətir uzunluğunu (79 simvol) keçməmək üçün iki yerə böldük
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    fact_n = 1
    for i in range(1, n + 1):
        fact_n *= i

    fact_x = 1
    for i in range(1, x + 1):
        fact_x *= i

    fact_nx = 1
    for i in range(1, (n - x) + 1):
        fact_nx *= i

    comb = fact_n / (fact_x * fact_nx)
    likelihood_values = comb * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values
