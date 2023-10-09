import numpy as np


class PlotterData:
    @staticmethod
    def get_x_plot_2d() -> np.array:
        _lim = 3
        _num = 50
        _ls = np.linspace(-_lim, _lim, _num)
        _x_0, _x_1 = np.meshgrid(_ls, _ls)
        _r = np.array([_x_0.flatten(), _x_1.flatten()])
        return _r
