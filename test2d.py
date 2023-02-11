import plotly.graph_objs as go
import numpy as np

# Crear datos iniciales
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Crear gráfico dinámico
fig = go.Figure(
    data=[go.Scatter(x=x, y=y)],
    layout=go.Layout(title="Gráfico dinámico de sin(x)")
)

# Actualizar gráfico en cada iteración
for i in range(2):
    y = np.sin(x + i / 10.0)
    fig.data[0].y = y
    fig.write_html("sin_plot.html", auto_open=True)
    fig.show()

# ejemplos de gráficos 3D

import plotly.graph_objects as go
import numpy as np

# Crear datos
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

# Crear gráfico 3D
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.show()

# graficar nubes de puntos en 2D

import plotly.graph_objects as go
import numpy as np

# Crear datos
N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Crear gráfico
fig = go.Figure(data=go.Scatter(x=random_x, y=random_y, mode='markers'))
fig.show()

# graficar dos grupos de nubes de puntos en 2D

import plotly.graph_objects as go
import numpy as np

# Crear datos
N = 1000
random_x = np.random.randn(N)
random_y0 = np.random.randn(N) + 5
random_y1 = np.random.randn(N)

# Crear gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=random_x, y=random_y0, mode='markers', name='markers'))
# color escala de temperatura
fig.add_trace(go.Scatter(x=random_x, y=random_y1, mode='markers', name='markers', marker_color='rgb(255, 0, 0)'))

fig.show()


# graficar nubes de puntos en 3D

import plotly.graph_objects as go
import numpy as np

# Crear datos
N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)
random_z = np.random.randn(N)

# Crear gráfico
fig = go.Figure(data=[go.Scatter3d(x=random_x, y=random_y, z=random_z, mode='markers')])
fig.show()
