import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearPolynomialModel(LinearModel):

    def __init__(self,
                 degree: int,  # degree of the polynomial (Number of factors)
                 init_weights: float | None = None,
                 ):
        super().__init__(degree, init_weights)
        # vector con los exponentes de cada factor
        self.nn: np.array = np.arange(self.f)

    def bb(self, x: float) -> np.array:
        return x ** self.nn
