import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearPolynomialModel(LinearModel):

    def __init__(self,
                 a: float,
                 f: int,  # factors_count (grado del polinomio)
                 init_weights_random: bool = True,
                 ):
        super().__init__(a, f, init_weights_random)
        # vector de los exponentes de cada factor
        self.nn: np.array = np.arange(self.f)

    def bb(self, xx: np.array) -> np.array:
        return xx ** self.nn
