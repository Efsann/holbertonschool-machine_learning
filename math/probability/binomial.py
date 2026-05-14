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

            # Ortalama və Dispersiyanı hesablayırıq
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # İlk olaraq p-ni təxmin edirik: p = 1 - (variance / mean)
            p_expected = 1 - (variance / mean)

            # n-i hesablayıb ən yaxın tam ədədə yuvarlaqlaşdırırıq
            self.n = round(mean / p_expected)

            # Yeni n-ə əsasən p-ni yenidən hesablayırıq ki, dəqiqlik itməsin
            self.p = float(mean / self.n)
