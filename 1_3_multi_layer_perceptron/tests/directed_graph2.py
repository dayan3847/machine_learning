import networkx as nx
import matplotlib.pyplot as plt

# Creamos el grafo
G = nx.Graph()

# Añadimos los nodos
G.add_nodes_from([1, 2, 3, 4])

# Añadimos las aristas con su peso
G.add_edge(1, 2, weight=0.5)
G.add_edge(1, 3, weight=1)
G.add_edge(2, 3, weight=2)
G.add_edge(2, 4, weight=1.5)
G.add_edge(3, 4, weight=0.5)

# Obtenemos la posición de los nodos en el plano
pos = nx.spring_layout(G)

# Obtenemos los pesos de las aristas para etiquetarlas
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

# Dibujamos el grafo con las etiquetas
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Mostramos la figura
plt.show()
