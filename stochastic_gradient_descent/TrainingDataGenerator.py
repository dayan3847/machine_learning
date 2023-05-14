import random
import numpy as np
import sympy as sp


# Training Data Generator
class TrainingDataGenerator:

    @staticmethod
    def generate_data_points(
            file_name: str,
            y_function: sp.Expr,  # target function
            x_interval: tuple = (0, 1),
            m: int = 100,  # Artificial Count
            epsilon_interval: tuple = (-.3, .3),
            x_uniform=True,  # False random
    ):
        x_list = np.arange(x_interval[0], x_interval[1], (x_interval[1] - x_interval[0]) / m) \
            if x_uniform else [random.uniform(x_interval[0], x_interval[1]) for _ in range(m)]
        y_list = []
        for x in x_list:
            e = random.uniform(epsilon_interval[0], epsilon_interval[1])
            y = y_function.subs('x', x) + e
            y_list.append(y)
        result: np.array = np.array([x_list, y_list]).T
        np.savetxt(file_name, result, delimiter=',')

        return result

    @staticmethod
    def generate_data_points_quadratic(file_name: str):
        x = sp.Symbol('x')
        function: sp.Expr = 5 + 10 * x + 5 * x ** 2
        return TrainingDataGenerator.generate_data_points(file_name, function), function

    @staticmethod
    def generate_data_points_cubic(file_name: str):
        x = sp.Symbol('x')
        function: sp.Expr = 5 + 0 * x - 9 * x ** 2 + 10 * x ** 3
        return TrainingDataGenerator.generate_data_points(file_name, function), function

    @staticmethod
    def generate_data_points_sinusoidal(file_name: str):
        x = sp.Symbol('x')
        function: sp.Expr = sp.sin(2 * sp.pi * x)
        return TrainingDataGenerator.generate_data_points(file_name, function), function
