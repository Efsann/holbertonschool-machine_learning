#!/usr/bin/env python3
"""
Module defining a single neuron with cost calculation
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
        - Private attribute __A (activated output)
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Parameters:
        - Y: numpy.ndarray with shape (1, m) containing correct labels
        - A: numpy.ndarray with shape (1, m) containing activated output

        Returns:
        - The calculated cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
