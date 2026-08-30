#!/usr/bin/env python3
"""
Module defining a single neuron with upgraded training capabilities
"""
import matplotlib.pyplot as plt
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
        - X: numpy.ndarray with shape (nx, m) containing input data

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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing input data
        - Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
        - prediction: numpy.ndarray with shape (1, m) with predicted labels
        - cost: calculated cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing input data
        - Y: numpy.ndarray with shape (1, m) containing correct labels
        - A: numpy.ndarray with shape (1, m) containing activated output
        - alpha: learning rate
        """
        m = Y.shape[1]
        dz = A - Y
        dw = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron with options for verbose logging and graphing

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing input data
        - Y: numpy.ndarray with shape (1, m) containing correct labels
        - iterations: number of iterations to train over
        - alpha: learning rate
        - verbose: boolean, whether to print training cost
        - graph: boolean, whether to plot training cost graph
        - step: step interval for verbose logging and graph points

        Returns:
        - Evaluation of training data after training
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        x_points = []

        for i in range(iterations + 1):
            if i == 0:
                A = self.forward_prop(X)
            else:
                self.gradient_descent(X, Y, self.__A, alpha)
                A = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)
                costs.append(current_cost)
                x_points.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(
                        i, current_cost))

        if graph:
            plt.plot(x_points, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
