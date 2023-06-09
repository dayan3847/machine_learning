import numpy as np
import plotly.graph_objs as go
from typing import List
from binary_classification.src.models import Artificial
from binary_classification.src.models import Polynomial
from binary_classification.src.tools import GrapherPlotlyData


class GrapherPlotlyData2D(GrapherPlotlyData):

    def __init__(self):
        super().__init__()
        self.figure.update_layout(
            xaxis_title="x0",
            yaxis_title="x1",
            xaxis_range=self.area_to_pot[0],
            yaxis_range=self.area_to_pot[1],
        )
        self.figure.add_trace(
            go.Scatter(
                x=[-10, 10],
                y=[0, 0],
                mode='lines',
                name='x',
                marker_color='gray'
            )
        )
        self.figure.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-10, 10],
                mode='lines',
                name='y',
                marker_color='gray',
            )
        )

    def plot_artificial_data(self, artificial: List[Artificial], name: str = 'Data', color: str = 'blue'):
        data_to_print: dict = {}
        for data in artificial:
            if data.y not in data_to_print:
                data_to_print[data.y] = {'x0': [], 'x1': []}
            data_to_print[data.y]['x0'].append(data.x_vector[0])
            data_to_print[data.y]['x1'].append(data.x_vector[1])
        for key in data_to_print:
            self.figure.add_trace(
                go.Scatter(
                    x=data_to_print[key]['x0'],
                    y=data_to_print[key]['x1'],
                    mode='markers',
                    name=f'{name} {key}',
                    marker_color=color,
                    opacity=.3 if 0 == key else 1,
                )
            )

    def plot_polynomial(self, polinomial: Polynomial, name: str, color: str = 'gray', y: int = .5):
        if polinomial.get_variables_count() > 2:
            return
        if polinomial.get_last_variable_degree() > 1:
            return
        x0_list = np.linspace(-3, 3, 100)
        x1_list = []
        for x0 in x0_list:
            x1 = polinomial.evaluate_despejando([x0], 1, y=y)
            x1_list.append(x1)
        self.figure.add_trace(
            go.Scatter(
                x=x0_list,
                y=x1_list,
                mode='lines',
                name=name,
                marker_color=color,
            )
        )

    def file_name(self) -> str:
        return 'figure_2d_data.html'
