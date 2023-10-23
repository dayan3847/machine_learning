import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Datos iniciales para los gráficos
x = np.arange(10)
y_data = np.random.rand(6, 10)  # 6 conjuntos de datos

# Configuración de la figura y los subplots (2 filas y 3 columnas)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Inicializa una lista para almacenar las líneas de los gráficos de línea
lines = []

# Crea los gráficos de línea iniciales y almacena las líneas en la lista
for i, ax in enumerate(axs.flat):
    line, = ax.plot(x, y_data[i])
    lines.append(line)


# Función de actualización de los gráficos
def update(frame):
    global y_data

    # Genera nuevos datos para los gráficos
    y_data = np.random.rand(6, 10)

    # Actualiza los datos de los gráficos de línea
    for i, line in enumerate(lines):
        line.set_ydata(y_data[i])

    return lines


# Crea la animación
ani = FuncAnimation(fig, update, blit=True, interval=1000)  # Actualiza cada segundo (1000 ms)

plt.show()
