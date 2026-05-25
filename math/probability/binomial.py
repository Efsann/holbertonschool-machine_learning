#!/usr/bin/env python3
"""Module that represents a Binomial distribution."""


class Binomial:
    """Class that represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor.

        Args:
            data: list of the data to be used to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_expected = 1 - (variance / mean)
            self.n = round(mean / p_expected)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Args:
            k: the number of successes

        Returns:
            The PMF value for k.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(x):
            res = 1
            for i in range(1, x + 1):
                res *= i
            return res

        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        pmf_value = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_value

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Args:
            k: the number of successes

        Returns:
            The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
