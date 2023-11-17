import numpy as np

from dayan3847.models.Model import Model


class LinearModel(Model):
    def __init__(self,
                 a: float,
                 f: int,  # factors_count
                 init_weights_random: bool = True,
                 ):
        self.a: float = a  # Learning rate
        self.f: int = f
        # Vector de pesos
        self.ww: np.array = np.random.rand(self.f) - .5 if init_weights_random \
            else np.zeros(self.f)

    def bb(self, xx: np.array) -> np.array:
        pass

    # Calculate the model value for a simple value
    def h(self, x: float) -> float:
        xx: np.array = np.full(self.f, x)
        bb: np.array = self.bb(xx)
        return self.ww @ bb

    @staticmethod
    def activate(h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    def g_single(self, x: float) -> float:
        return self.activate(self.h(x))

    def g_set(self, x_set: np.array) -> np.array:
        return np.array([self.g_single(x) for x in x_set])

    def update_w_single(self, x: float, y: float):
        xx: np.array = np.full(self.f, x)
        yy: np.array = np.full(self.f, y)
        bb: np.array = self.bb(xx)
        h: float = self.ww @ bb
        g: float = self.activate(h)
        a: float = self.a

        self.ww += a * (yy - g) * bb

    def g(self, x):
        return self.g_set(x) if isinstance(x, np.ndarray) \
            else self.g_single(x)
