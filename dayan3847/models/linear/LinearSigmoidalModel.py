import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearSigmoidalModel(LinearModel):

    def __init__(self,
                 sigmoidal: tuple[
                     float,  # min
                     float,  # max
                     int,  # number of sigmoidal
                 ],
                 s: float,  # slope
                 init_weights: float | None = None,
                 ):
        f = sigmoidal[2]
        super().__init__(f, init_weights)
        self.mm: np.array = np.linspace(*sigmoidal)
        self.s: float = s

    def bb(self, x: float) -> np.array:
        mm: np.array = self.mm
        s: float = self.s
        mm_xx: np.array = mm - x
        return 1 / (1 + np.exp(mm_xx / s))
