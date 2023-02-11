# graficar nubes de puntos en 3D

import plotly.graph_objects as go
import numpy as np

# Crear datos
N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)
random_z = np.random.randn(N)

# Crear gráfico
fig = go.Figure()
## altura en escala de temperatura
fig.add_trace(go.Scatter3d(x=random_x, y=random_y, z=random_z, mode='markers',
                           marker=dict(size=12, color=random_z, colorscale='Viridis', opacity=0.8)))
fig.show()

# graficar un plano en 3D

import plotly.graph_objects as go
import numpy as np

# Crear datos

# Crear gráfico
fig = go.Figure(data=[go.Surface(z=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])])
fig.show()

# graficar un hiperplano en 3D

import plotly.graph_objects as go
import numpy as np

# Crear datos
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2

# Crear gráfico
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.show()
