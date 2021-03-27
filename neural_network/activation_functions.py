import numpy as np


class Identity:

    @staticmethod
    def calculate(x):
        return x

    @staticmethod
    def calculate_derivative(x):
        return 1


class HyperbolicTangent:

    @staticmethod
    def calculate(x):
        return np.tanh(x)

    @staticmethod
    def calculate_derivative(x):
        return 1 - np.tanh(x) ** 2


class Sigmoid:

    @staticmethod
    def calculate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def calculate_derivative(x):
        return Sigmoid.calculate(x) * (1 - Sigmoid.calculate(x))


class RectifiedLinearUnit:

    @staticmethod
    def calculate(x):
        return np.maximum(0, x)

    @staticmethod
    def calculate_derivative(x):
        return np.heaviside(x, 0)
