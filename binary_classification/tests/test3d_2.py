import plotly.express as px
import numpy as np

# Crear datos
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Crear gráfico 3D
# fig = px.surface(x=X, y=Y, z=Z)
fig = px.line_3d(x=X, y=Y, z=Z)

fig.show()
