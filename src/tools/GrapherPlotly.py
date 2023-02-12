import plotly.graph_objs as go
from abc import ABC, abstractmethod
from plotly.graph_objs import Figure


class GrapherPlotly(ABC):
    figure: Figure

    def __init__(self):
        self.figure = go.Figure()

    def show(self):
        self.figure.show()

    def save(self, path: str = './'):
        name: str = path + self.file_name()
        self.figure.write_html(name, auto_open=True)

    @abstractmethod
    def file_name(self) -> str:
        pass
