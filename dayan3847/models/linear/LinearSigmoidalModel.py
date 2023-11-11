import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearSigmoidalModel(LinearModel):

    def __init__(self,
                 a: float,
                 f: int,
                 mu_lim: tuple[float, float],  # mu_limits
                 s: float,
                 init_weights_random: bool = True,
                 ):
        super().__init__(a, f, init_weights_random)
        self.mm: np.array = np.linspace(mu_lim[0], mu_lim[1], self.f)
        self.s: float = s

    def bb(self, xx: np.array) -> np.array:
        mm: np.array = self.mm
        s: float = self.s
        mm_xx: np.array = mm - xx
        return 1 / (1 + np.exp(mm_xx / s))
