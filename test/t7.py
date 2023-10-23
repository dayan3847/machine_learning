import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Datos iniciales para el gráfico de barras y el gráfico de puntos
x = np.arange(10)
y_bar = np.random.rand(10)
y_scatter = np.random.rand(10)

# Configuración de la figura y los subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Gráfico de barras inicial
bars = ax1.bar(x, y_bar)

# Gráfico de puntos inicial
scatter = ax2.scatter(x, y_scatter)


# Función de actualización de los gráficos
def update(frame):
    global y_bar, y_scatter

    # Genera nuevos datos para los gráficos
    y_bar = np.random.rand(10)
    y_scatter = np.random.rand(10)

    # Actualiza los datos de los gráficos
    for bar, y in zip(bars, y_bar):
        bar.set_height(y)

    scatter.set_offsets(np.column_stack((x, y_scatter)))

    # return bars, scatter


# Crea la animación
ani = FuncAnimation(fig, update, interval=1000)  # Actualiza cada segundo (1000 ms)

plt.show()
