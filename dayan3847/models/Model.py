import numpy as np


class Model:

    def g(self, x):
        pass

    def update_w(self, x, y):
        return self.update_w_set(x, y) if isinstance(y, np.ndarray) \
            else self.update_w_single(x, y)

    def update_w_set(self, x_set: np.array, y_set: np.array):
        for x, y in zip(x_set, y_set):
            self.update_w_single(x, y)

    def update_w_single(self, x, y):
        pass

    def data_to_plot_matplotlib(self, num=20) -> np.array:
        pass
