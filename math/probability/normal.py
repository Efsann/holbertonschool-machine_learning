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

            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x: the x-value

        Returns:
            The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z: the z-score

        Returns:
            The x-value of z.
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Args:
            x: the x-value

        Returns:
            The PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536

        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * (((x - self.mean) / self.stddev) ** 2)
        pdf_value = coefficient * (e ** exponent)

        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Args:
            x: the x-value

        Returns:
            The CDF value for x.
        """
        pi = 3.1415926536

        # 1. z dəyərini tapırıq
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # 2. Xəta funksiyasını (erf) Taylor sırası ilə hesablayırıq
        erf_z = (2 / (pi ** 0.5)) * (z - (z ** 3) / 3 + (z ** 5) / 10 -
                                     (z ** 7) / 42 + (z ** 9) / 216)

        # 3. CDF düsturunu tətbiq edirik
        cdf_value = 0.5 * (1 + erf_z)

        return cdf_value
