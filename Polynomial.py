import random
from typing import List

from Factor import Factor


class Polynomial:
    # Factors of the polynomial
    factors: List[Factor]
    # Coefficients of the polynomial
    thetas: List[float]
    # Number of terms of the polynomial
    n_terms: int
    # Number of variables of the polynomial
    n_variables: int

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
        self.n_terms = len_factors
        self.n_variables = 0
        for factor in factors:
            if factor.variable > self.n_variables:
                self.n_variables = factor.variable
        self.n_variables += 1

    # evaluate the polynomial
    def evaluate(self, x_list: List[float]) -> float:
        if len(x_list) != self.n_variables:
            raise ValueError("x_list and the polynomial must have the same length")
        result: float = 0
        for i in range(self.n_terms):
            theta: float = self.thetas[i]
            if 0 == theta:
                continue
            x: float = x_list[self.factors[i].variable]
            result += theta * x ** self.factors[i].degree
        return result

    # evaluate the polynomial despejando una variable xk
    def evaluate_despejando(self, x_list: List[float], k: int, y: float = 0) -> float:
        result: float = 0
        theta_xk: float | None = None
        for i in range(len(self.factors)):
            factor: Factor = self.factors[i]
            if k == factor.variable:
                if theta_xk is not None:
                    raise ValueError("The variable is repeated")
                if 1 != factor.degree:
                    raise ValueError("The variable is not in the first degree")
                theta_xk = self.thetas[i]
                if 0 == theta_xk:
                    raise ValueError("No se puede despejar una variable con theta = 0")
                continue
            theta: float = self.thetas[i]
            if 0 == theta:
                continue
            x: float = x_list[factor.variable]
            result += theta * x ** self.factors[i].degree
        return (y - result) / theta_xk

    # get number of independent variables
    def get_variables_count(self) -> int:
        variables_count: int = 0
        for factor in self.factors:
            if factor.variable >= variables_count:
                variables_count = factor.variable + 1
        return variables_count

    # get the last factor of highest degree
    def get_last_factor(self) -> Factor:
        last_factor: Factor = self.factors[0]
        for factor in self.factors[1:]:
            if factor.variable > last_factor.variable or (
                    factor.variable == last_factor.variable and factor.degree > last_factor.degree
            ):
                last_factor = factor
        return last_factor

    def get_last_variable_degree(self) -> int:
        last_factor: Factor = self.get_last_factor()
        return last_factor.degree
