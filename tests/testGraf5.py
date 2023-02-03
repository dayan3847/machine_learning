import plotly.graph_objs as go
import numpy as np

# Crear datos iniciales
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Crear gráfico dinámico
fig = go.Figure(
    data=[go.Scatter(x=x, y=y)],
    layout=go.Layout(title="Gráfico dinámico de sin(x)")
)

# Actualizar gráfico en cada iteración
for i in range(3):
    y = np.sin(x + i/10.0)
    fig.data[0].y = y
    # fig.write_html("sin_plot.html", auto_open=True)
    fig.show()
