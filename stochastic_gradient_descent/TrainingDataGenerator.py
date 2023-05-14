import random
import numpy as np
import sympy as sp


# Training Data Generator
class TrainingDataGenerator:

    @staticmethod
    def generate_data_points(
            path: str,
            y_function: sp.Expr,  # target function
            data_points_file_name: str = 'data_points.csv',
            base_function_file_name: str = 'data_points_base_function.txt',
            x_interval: tuple = (0, 1),
            m: int = 100,  # Artificial Count
            epsilon_interval: tuple = (-.3, .3),
            x_uniform=True,  # False random
    ) -> np.array:
        x_list = np.linspace(0, 1, 100) if x_uniform \
            else [random.uniform(x_interval[0], x_interval[1]) for _ in range(m)]
        y_list = []
        for x in x_list:
            e = random.uniform(epsilon_interval[0], epsilon_interval[1])
            y = y_function.subs('x', x) + e
            y_list.append(y)
        result: np.array = np.array([x_list, y_list])
        np.savetxt(f'{path}/{data_points_file_name}', result.T, delimiter=',')
        y_function_string: str = str(y_function)
        with open(f'{path}/{base_function_file_name}', 'w') as f:
            f.write(y_function_string)

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
