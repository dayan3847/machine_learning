from abc import ABC, abstractmethod
from Polynomial import Polynomial
import numpy as np


class GrapherPlotly(ABC):

    def __init__(self):
        # self.area_to_pot = ((-3, 3), (-7, 3))
        self.area_to_pot = ((-5, 5), (-10, 10), (-.1, 1.1))
        self.data_to_pot = {
            'x0': np.linspace(self.area_to_pot[0][0], self.area_to_pot[0][1], 50),
            'x1': np.linspace(self.area_to_pot[1][0], self.area_to_pot[1][1], 50),
        }
        self.data_to_pot['x0'], self.data_to_pot['x1'] = np.meshgrid(
            self.data_to_pot['x0'], self.data_to_pot['x1']
        )

    # abstract method

    @abstractmethod
    def plot_polynomial(self, polinomial: Polynomial, name: str, color: str = 'gray', show: bool = False):
        pass
