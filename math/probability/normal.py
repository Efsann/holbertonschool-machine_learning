#!/usr/bin/env python3
"""Module that represents a Normal distribution."""


class Normal:
    """Class that represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor.

        Args:
            data: list of the data to be used to estimate the distribution
            mean: the mean of the distribution
            stddev: the standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Ortalamanın (mean) hesablanması
            self.mean = float(sum(data) / len(data))

            # Dispersiya (variance) və Standart meylin (stddev) hesablanması
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
