import numpy as np
from typing import List
from abc import abstractmethod
from binary_classification.src.models import Polynomial, Artificial
from binary_classification.src.tools import GrapherPlotly


class GrapherPlotlyData(GrapherPlotly):

    def __init__(self):
        super().__init__()

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
    def plot_polynomial(self, polinomial: Polynomial, name: str, color: str = 'gray'):
        pass

    @abstractmethod
    def plot_artificial_data(self, artificial: List[Artificial], name: str = 'Data', color: str = 'blue'):
        pass
