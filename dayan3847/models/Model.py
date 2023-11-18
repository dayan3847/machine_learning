import numpy as np

from dayan3847.tools.functions import check_vector_dim


class Model:

    def __init__(self,
                 f: int,  # Number of factors
                 init_weights: float | None = None,  # None: random, float: init weights with this value
                 ):
        if f == 0:
            raise Exception('f invalid')

        self.f: int = f
        # Weights Vector:
        self.ww: np.array = np.random.rand(f) - .5 if init_weights is None \
            else np.full(f, init_weights)

    def g(self, x):
        pass

    def g_set(self, x_set) -> np.array:
        return np.array([self.g_single(x) for x in x_set])

    def g_single(self, x) -> float:
        return self.activate(self.h(x))

    def h(self, x) -> float:
        bb: np.array = self.bb(x)
        check_vector_dim(bb, self.f)
        return self.ww @ bb

    def bb(self, x) -> np.array:
        pass

    @staticmethod
    def activate(h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    def train(self, x, y, a: float):
        return self.train_set(x, y, a) if isinstance(y, np.ndarray) \
            else self.train_single(x, y, a)

    def train_set(self, x_set: np.array, y_set: np.array, a: float):
        for x, y in zip(x_set, y_set):
            self.train_single(x, y, a)

    def train_single(self, x, y, a: float):
        x = np.array(x)
        bb: np.array = self.bb(x)
        h: float = self.ww @ bb
        g: float = self.activate(h)

        self.ww += a * (y - g) * bb

    def data_to_plot_matplotlib(self, num=20) -> np.array:
        pass

    def get_ww(self) -> np.array:
        return self.ww

    def set_ww(self, ww: np.array):
        check_vector_dim(ww, self.f)
        self.ww = ww
