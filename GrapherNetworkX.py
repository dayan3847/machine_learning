from typing import List

import networkx as nx
import matplotlib.pyplot as plt

from Network import Network
from Perceptron import Perceptron


class GrapherNetworkX:

    @staticmethod
    def draw_network(network: Network, show: bool = True):
        graph: nx.DiGraph = nx.DiGraph()
        features_count: int = len(network.neurons[0][0].weight_features)
        for i in range(features_count):
            graph.add_node(f'0{i}', pos=(0, -i), label=f'f{i}')

        for i in range(len(network.neurons)):
            i_layer: List[Perceptron] = network.neurons[i]
            for j in range(len(i_layer)):
                graph.add_node(f'{i + 1}{j}', pos=(i + 1, -j), label=f'n{i + 1}{j}')
                j_neuron: Perceptron = i_layer[j]
                for k in range(len(j_neuron.weight_features)):
                    k_weight = j_neuron.weight_features[k]
                    graph.add_edge(f'{i}{k}', f'{i + 1}{j}', weight=k_weight)

        graph.add_edge(f'00', f'10', weight=5)

        nx.draw(
            graph,
            with_labels=True,
            font_weight='bold',
            node_size=1000,
            node_color='green',
            pos=nx.get_node_attributes(graph, 'pos'),
        )
        plt.show()
