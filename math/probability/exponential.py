#!/usr/bin/env python3
"""Module that represents an Exponential distribution."""


class Exponential:
    """Class that represents an Exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor.

        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Args:
            x: the time period

        Returns:
            The PDF value for x, or 0 if x is out of range.
        """
        if x < 0:
            return 0

        e = 2.7182818285
        pdf_value = self.lambtha * (e ** (-self.lambtha * x))

        return pdf_value
