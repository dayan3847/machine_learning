import numpy as np

from dayan3847.models.linear.LinearModel import LinearModel


class LinearGaussianModel(LinearModel):

    def __init__(self,
                 gaussian: tuple[
                     float,  # min
                     float,  # max
                     int,  # number of sigmoidal
                     float,  # s2
                 ],
                 init_weights: float | None = None,
                 ):
        f = gaussian[2]
        super().__init__(f, init_weights)

        self.mm: np.array = np.linspace(*gaussian[:3])
        self.s2: float = gaussian[3]

    def bb(self, xx: np.array) -> np.array:
        mm: np.array = self.mm
        s2: float = self.s2
        xx_mm: np.array = xx - mm
        xx_mm_2: np.array = xx_mm ** 2
        return np.exp(-.5 * xx_mm_2 / s2)
