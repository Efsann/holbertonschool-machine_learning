#!/usr/bin/env python3
"""Module for computing Bayesian Probability Marginal."""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.

    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: a 1D numpy.ndarray containing the various probabilities
        Pr: a 1D numpy.ndarray containing the prior beliefs about P

    Returns:
        The marginal probability of obtaining x and n.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Likelihood hesablanması
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

    # Intersection hesablanması (Likelihood * Prior)
    intersection_values = likelihood_values * Pr

    # Marginal ehtimal (Bütün kəsişmələrin cəmi)
    marginal_prob = np.sum(intersection_values)

    return marginal_prob
