import numpy as np


class PlotterData:

    # the data would be in a DxN matrix
    # where D is the dimension of the data and N is the number of data
    @staticmethod
    def get_x_plot_2d(lim_=(0, 1)) -> np.array:
        _num = 50
        _ls = np.linspace(lim_[0], lim_[1], _num)
        _x_0, _x_1 = np.meshgrid(_ls, _ls)
        _r = np.array([_x_0.flatten(), _x_1.flatten()])
        return _r
