from networkx import DiGraph


class Neuron:
    _graph: DiGraph
    _value: float | None

    def __init__(self, graph: DiGraph):
        self._graph = graph

    def get_value(self) -> float:
        pass

    def set_value(self, value: float = None):
        self._value = value

    def __str__(self):
        return self._value if self._value is not None else 'None value'
