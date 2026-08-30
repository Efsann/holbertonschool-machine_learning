#!/usr/bin/env python3
"""
Module defining a neural network with private attributes
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Class constructor for NeuralNetwork

        Parameters:
        - nx: int, number of input features
        - nodes: int, number of nodes found in the hidden layer

        Raises:
        - TypeError: if nx or nodes is not an integer
        - ValueError: if nx or nodes is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for __W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for __b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for __A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for __W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for __b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for __A2"""
        return self.__A2
