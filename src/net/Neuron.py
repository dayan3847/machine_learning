from networkx import DiGraph


class Neuron:
    _graph: DiGraph
    _value: float | None

    def __init__(self, graph: DiGraph, value: float = None):
        self._graph = graph
        self._value = value

    def get_value(self) -> float:
        pass

    def clear_value(self):
        self._value = None

    def __str__(self):
        return self._value if self._value is not None else 'None value'
