import numpy as np
import plotly.graph_objs as go
from typing import List
from src.models import Artificial
from src.models import Polynomial
from src.tools import GrapherPlotlyData


class GrapherPlotlyData3D(GrapherPlotlyData):

    def __init__(self):
        super().__init__()
        self.figure.update_layout(
            scene=dict(
                xaxis_title='x0',
                yaxis_title='x1',
                zaxis_title='y',
                xaxis_range=self.area_to_pot[0],
                yaxis_range=self.area_to_pot[1],
                zaxis_range=self.area_to_pot[2],
            ),
            showlegend=True,
        )

    def plot_artificial_data(self, artificial: List[Artificial], name: str = 'Data', color: str = 'blue'):
        x0_list_y0 = []
        x1_list_y0 = []
        x0_list_y1 = []
        x1_list_y1 = []
        for data in artificial:
            if data.y == 0:
                x0_list_y0.append(data.x_vector[0])
                x1_list_y0.append(data.x_vector[1])
            else:
                x0_list_y1.append(data.x_vector[0])
                x1_list_y1.append(data.x_vector[1])
        self.figure.add_trace(
            go.Scatter3d(
                x=x0_list_y0,
                y=x1_list_y0,
                z=[0] * len(x0_list_y0),
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=.3,
                ),
                name=f'{name} 0',
            )
        )
        self.figure.add_trace(
            go.Scatter3d(
                x=x0_list_y1,
                y=x1_list_y1,
                z=[1] * len(x0_list_y1),
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=1,
                ),
                name=f'{name} 1',
            )
        )

    def plot_polynomial(self, polinomial: Polynomial, name: str, color: str = 'Grays'):
        if polinomial.get_variables_count() > 3:
            return
        x0 = self.data_to_pot['x0']
        x1 = self.data_to_pot['x1']
        y = polinomial.evaluate([x0, x1])
        self.figure.add_trace(
            go.Surface(
                z=y,
                x=x0,
                y=x1,
                opacity=0.5,
                showscale=False,
                colorscale=color,
                name=name,
                showlegend=True,
            )
        )

    # graficar un plano y
    def plot_plane_y(self, y: float):
        x0 = self.data_to_pot['x0']
        x1 = self.data_to_pot['x1']
        y = np.full((len(x0), len(x1)), y)

        self.figure.add_trace(
            go.Surface(
                z=y,
                x=x0,
                y=x1,
                opacity=.3,
                showscale=False,
                colorscale='Greys',
                name='Plane y = 0',
                showlegend=True,
            )
        )

    def file_name(self) -> str:
        return 'figure_3d_data.html'
