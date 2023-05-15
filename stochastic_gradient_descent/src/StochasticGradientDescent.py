import random
import time
import numpy as np
import sympy as sp


class StochasticGradientDescent:

    def __init__(
            self,
            training_data: np.array,  # Training Data
            d: int = 10,  # Polynomial Degree
            a: float = .001,  # Alpha
            iterations_count: int = 1000,  # Iterations Count
    ):
        self.data_point = training_data

        # Artificial Count
        self.m: int = len(self.data_point[0])
        # Polynomial Degree
        self.d: int = d
        # Alpha
        self.a: float = a
        # Iterations Count
        self.iterations_count: int = iterations_count

        # Generate Initial Parameters of the polynomial
        ipi: tuple = (-.5, .5)  # initial_parameters_interval
        self.parameters: list = []
        for i in range(self.d + 1):
            parameter = random.uniform(ipi[0], ipi[1])
            self.parameters.append(parameter)

        polynomial_initial: sp.Expr = self.get_polynomial()
        error_initial: float = self.get_error()
        self.polynomial_best: sp.Expr = polynomial_initial
        self.error_best: float = error_initial
        self.history = {
            'interation': [0],
            'polynomial': [polynomial_initial],
            'error': [error_initial],
            'polynomial_best': [self.polynomial_best],
            'error_best': [self.error_best],
        }

    def get_polynomial(self) -> sp.Expr:
        polynomial_str = ''
        for i in range(self.d + 1):
            polynomial_str += f'{self.parameters[i]} * x ** {i} + '

        return sp.sympify(polynomial_str[:-3])

    def get_error(self) -> float:
        error: float = 0
        for i in range(self.m):
            xi = self.data_point[0][i]
            yi = self.data_point[1][i]
            hi = self.h(xi)
            error += (hi - yi) ** 2
        return error / 2

    def run(self) -> float:
        init_time: float = time.time()
        for k in range(self.iterations_count):
            for i in range(self.m):
                xi = self.data_point[0][i]
                yi = self.data_point[1][i]
                for j in range(self.d + 1):
                    hi = self.h(xi)
                    self.parameters[j] += self.a * (yi - hi) * (xi ** j)
            ek = self.get_error_rms()
            if ek < self.error_best:
                self.error_best = ek
                self.polynomial_best = self.get_polynomial()
            self.save_history(k + 1)
        end_time: float = time.time()
        return end_time - init_time

    def h(self, x: float) -> float:
        result = 0
        for i in range(self.d + 1):
            result += self.parameters[i] * (x ** i)
        return result

    def get_error_rms(self) -> float:
        e: float = self.get_error()
        return (2 * e / self.m) ** 0.5

    def save_history(self, iteration):
        # Save History
        self.history['interation'].append(iteration)
        self.history['polynomial'].append(self.get_polynomial())
        self.history['error'].append(self.get_error())
        self.history['polynomial_best'].append(self.polynomial_best)
        self.history['error_best'].append(self.error_best)
