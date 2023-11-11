import numpy as np
import networkx as nx
from networkx import DiGraph
from matplotlib import pyplot as plt
from typing import List

from dayan3847.multi_layer_perceptron.src.net.Neuron import Neuron
from dayan3847.multi_layer_perceptron.src.net.NeuronInput import NeuronInput
from dayan3847.multi_layer_perceptron.src.net.NeuronOutput import NeuronOutput
from dayan3847.multi_layer_perceptron.src.net.NeuronPerceptron import NeuronPerceptron


class Network:
    _graph: DiGraph
    _neurons: List[List[Neuron]]
    _neurons_input: List[NeuronInput]  # _neurons[0]
    _neurons_output: List[NeuronOutput]  # _neurons[-1]

    def __init__(self, topology: List[int]):
        topology_len: int = len(topology)
        if topology_len < 2:
            raise Exception("Network must have at least 2 layers")
        self._graph = DiGraph()
        self._neurons = []
        # Input Neuron:
        self._neurons_input = []
        input_layer_count: int = topology[0]
        for i in range(input_layer_count):
            node_input: NeuronInput = NeuronInput(self._graph)
            self._graph.add_node(node_input, pos=(0, -i))
            self._neurons_input.append(node_input)
        self._neurons.append(self._neurons_input)
        # Hidden Neuron:
        for t in range(1, topology_len - 1):
            i_layer_count = topology[t]
            neurons_layer_i = []
            for i in range(i_layer_count):
                neuron: NeuronPerceptron = NeuronPerceptron(self._graph, self.generate_weight())
                self._graph.add_node(neuron, pos=(t, -i))
                neurons_layer_i.append(neuron)
                self.connect_neuron_with_layer(neuron, i - 1)
            self._neurons.append(neurons_layer_i)
        # Output Neuron:
        self._neurons_output = []
        output_layer_count: int = topology[-1]
        for i in range(output_layer_count):
            neuron: NeuronOutput = NeuronOutput(self._graph, self.generate_weight())
            self._graph.add_node(neuron, pos=(topology_len - 1, -i))
            self._neurons_output.append(neuron)
            self.connect_neuron_with_layer(neuron, -1)

    def propagate(self, features: List[float]) -> List[float]:
        count: int = len(features)
        if count != len(self._neurons_input):
            raise Exception('Error')
        # Clear neurons values:
        for layer in self._neurons:
            for neuron in layer:
                neuron.clear_value()
        # Set network input values:
        for i in range(count):
            f: float = features[i]
            neuron: NeuronInput = self._neurons_input[i]
            neuron.set_value(f)

        return [o_neuron.get_value() for o_neuron in self._neurons_output]

    def generate_weight(self) -> float:
        return np.random.uniform(-.5, .5)

    def generate_weights(self, count: int) -> List[float]:
        return [self.generate_weight() for _ in range(count)]

    def connect_neuron_with_layer(self, neuron: Neuron, layer_index: int):
        for neuron_prev in self._neurons[layer_index]:
            self._graph.add_edge(neuron_prev, neuron, weight=self.generate_weight())

    def draw(self):
        nx.draw(
            self._graph,
            with_labels=True,
            font_weight='bold',
            node_size=1000,
            node_color='green',
            pos=nx.get_node_attributes(self._graph, 'pos'),
        )
        plt.show()
