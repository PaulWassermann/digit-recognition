import numpy as np


class AbsoluteValue:

    @staticmethod
    def calculate(x, y):
        return np.abs(y - x)

    @staticmethod
    def calculate_derivative(x, y):
        return np.sign(y - x)


class SquaredError:

    @staticmethod
    def calculate(x, y):
        return (1 / 2) * (y - x) ** 2

    @staticmethod
    def calculate_derivative(x, y):
        return x - y


class CrossEntropy:

    @staticmethod
    def calculate(x, y):
        return - np.sum(np.nan_to_num(y * np.log(x) + (1 - y) * np.log(1 - x)))

    @staticmethod
    def calculate_derivative(x, y):
        return np.sum(np.nan_to_num((x - y) / (x * (1 - x))))
