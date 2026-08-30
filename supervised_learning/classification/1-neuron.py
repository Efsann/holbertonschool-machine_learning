#!/usr/bin/env python3
"""
Module defining a single neuron with private attributes
"""
import numpy as np


class Neuron:
    """
    Class Neuron that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Class constructor for Neuron

        Parameters:
        - nx: int, number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for __W"""
        return self.__W

    @property
    def b(self):
        """Getter for __b"""
        return self.__b

    @property
    def A(self):
        """Getter for __A"""
        return self.__A
