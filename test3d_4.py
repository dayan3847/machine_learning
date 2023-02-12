import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Crear datos para el primer plano
x1, y1 = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
z1 = 2 * x1 + 3 * y1

# Crear datos para el segundo plano
x2, y2 = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
z2 = -2 * x2 - 3 * y2

# Crear el gráfico 3D
fig = go.Figure(data=[go.Surface(x=x1, y=y1, z=z1, showscale=False, opacity=0.7),
                     go.Surface(x=x2, y=y2, z=z2, showscale=False, opacity=0.7),
                     go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='black'))])

# Definir los límites del gráfico
fig.update_layout(scene=dict(xaxis_range=[-10, 10], yaxis_range=[-10, 10], zaxis_range=[-30, 30]))

# Mostrar el gráfico
fig.show()


# graficar una linea en 3D con plotly

import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Crear datos
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
z = 2 * x + 3 * y

# Crear el gráfico 3D
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=2))])

