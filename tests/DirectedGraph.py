import networkx as nx
from networkx import Graph, DiGraph

graph: Graph = DiGraph()
graph.add_node(1, pos=(0, -1))
graph.add_node(2, pos=(0, -2))
graph.add_node(3, pos=(0, -3))

graph.add_node(11, pos=(1, -1))
graph.add_node(12, pos=(1, -2))
graph.add_node(13, pos=(1, -3))

# agregar arista entre 1 y 11 con label 1
graph.add_edge(1, 11, text=1, label=1, color='red', width=2, style='dashed', arrowhead='vee', arrowsize=20)
graph.add_edge(1, 12, label=2)
graph.add_edge(1, 13, label=3)
graph.add_edge(1, 2, weight=3)
graph.add_edge(1, 3, weight=7, capacity=15, length=342.7)

# graph.add_edge(2, 11)
# graph.add_edge(2, 12)
# graph.add_edge(2, 13)
#
# graph.add_edge(3, 11, weight=2, label='A', color='red', width=2, style='dashed', arrowhead='vee', arrowsize=20)
# graph.add_edge(3, 12)
# graph.add_edge(3, 13)

# graficar
import matplotlib.pyplot as plt

plt.subplot(111)
plt.title("Grafo dirigido")
nx.draw(
    graph,
    with_labels=True,
    font_weight='bold',
    node_size=1000,
    node_color='green',
    pos=nx.get_node_attributes(graph, 'pos'),
)

plt.show()

# graficar con plotly

import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=[1, 2, 3, 11, 12, 13],
            y=[1, 2, 3, 11, 12, 13],
            mode="markers+text",
            marker=dict(
                size=50,
                color='green',
                opacity=1,
            ),
            name='Nodos',
            text=['A', 'B', 'C', 'D', 'E', 'F'],
        ),
        go.Scatter(
            x=[1, 1, None, 1, 2, 2, 2, 3, 3, 3],
            y=[11, 12, None, 13, 11, 12, 13, 11, 12, 13],
            mode='lines+text',
            line=dict(
                width=1,
                color='gray',
            ),
            name='Aristas',
            text=['A', 'B', 'C', 'D', 'E', 'F'],
            showlegend=True,
            textposition="bottom center",
        ),
    ],
)

fig.show()

# from dash import Dash, html
# import dash_cytoscape as cyto
#
# app = Dash(__name__)
#
# app.layout = html.Div([
#     html.P("Dash Cytoscape:"),
#     cyto.Cytoscape(
#         id='cytoscape',
#         elements=[
#             {'data': {'id': 'ca', 'label': 'Canada'}},
#             {'data': {'id': 'ca2', 'label': 'Canada2'}},
#             {'data': {'id': 'on', 'label': 'Ontario'}},
#             {'data': {'id': 'qc', 'label': 'Quebec'}},
#
#             {'data': {'source': 'ca', 'target': 'on', 'label': 'border'}},
#             {'data': {'source': 'ca', 'target': 'qc', 'label': 'border'}},
#             {'data': {'source': 'ca2', 'target': 'on', 'label': 'border'}},
#             {'data': {'source': 'ca2', 'target': 'qc', 'label': 'border'}},
#         ],
#         layout={'name': 'breadthfirst'},
#         style={'width': '400px', 'height': '500px'}
#     )
# ])
#
# app.run_server()
import plotly.graph_objs as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], mode="markers+text+lines", name="My Line 1-2",
                         text=["My Point", "My Point2"], textposition="bottom center"))
fig.add_trace(go.Scatter(x=[2], y=[2], mode="markers+text", name="My Point2",
                         text=["My Point2"], textposition="bottom center"))

fig.show()


import plotly.graph_objs as go

# Define los nodos
nodes = go.Scatter(x=[0, 1, 2, 3], y=[0, 1, 2, 3], mode='markers',
                   marker=dict(size=20, color='red'),
                   text=["A", "B", "C", "D"])

# Define las aristas
edges = go.Scatter(x=[0, 1, 1, 2], y=[1, 0, 2, 1], mode='lines',
                   line=dict(color='black', width=2),
                   hoverinfo='none')

# Define las etiquetas de los arcos
edge_labels = go.Scatter(x=[0.5, 1, 1.5], y=[0.5, 1.5, 1.5], mode='text',
                         text=['Edge 1', 'Edge 2', 'Edge 3'],
                         textfont=dict(color='black', size=16))

# Crea la figura
fig = go.Figure(data=[edges, nodes, edge_labels],
                layout=go.Layout(title='Ejemplo de gráfico de red con etiquetas'))

# Muestra la figura
fig.show()
