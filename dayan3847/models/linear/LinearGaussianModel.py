import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearGaussianModel(LinearModel):

    def __init__(self,
                 a: float,
                 f: int,
                 mu_lim: tuple[float, float],  # mu_limits
                 s: float,
                 init_weights_random: bool = True,
                 ):
        super().__init__(a, f, init_weights_random)
        self.mm: np.array = np.linspace(mu_lim[0], mu_lim[1], self.f)
        self.s2: float = s ** 2

    def bb(self, xx: np.array) -> np.array:
        mm: np.array = self.mm
        s2: float = self.s2
        xx_mm: np.array = xx - mm
        xx_mm_2: np.array = xx_mm ** 2
        return np.exp(-.5 * xx_mm_2 / s2)
