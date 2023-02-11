import random
from typing import List

from Factor import Factor


class Polynomial:
    # Factors of the polynomial
    factors: List[Factor]
    # Coefficients of the polynomial
    thetas: List[float]

    # init
    def __init__(self, factors: List[Factor] = None, thetas=None):
        if factors is None:
            factors = [Factor()]
        if thetas is None:
            thetas = []
        len_thetas: int = len(thetas)
        len_factors: int = len(factors)
        if 0 < len_thetas and len(factors) != len_thetas:
            raise ValueError("factors and thetas must have the same length")
        self.factors = factors
        if 0 == len_thetas:
            for i in range(len_factors):
                thetas.append(1)
        self.thetas = thetas

    # evaluate the polynomial
    def evaluate(self, x_list: List[float]) -> float:
        result: float = 0
        for i in range(len(self.factors)):
            theta: float = self.thetas[i]
            if 0 == theta:
                continue
            x: float = x_list[self.factors[i].variable]
            result += theta * x ** self.factors[i].degree
        return result

    # initialize the thetas randomly
    def init_thetas(self, thetas_range: (float, float)):
        for i in range(len(self.thetas)):
            self.thetas[i] = random.uniform(thetas_range[0], thetas_range[1])

    # get number of independent variables
    def get_variables_count(self) -> int:
        variables_count: int = 0
        for factor in self.factors:
            if factor.variable >= variables_count:
                variables_count = factor.variable + 1
        return variables_count
