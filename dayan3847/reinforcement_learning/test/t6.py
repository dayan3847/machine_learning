import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Inicializa los datos
categorias = ['Categoría 1', 'Categoría 2', 'Categoría 3', 'Categoría 4']
valores = [10, 25, 15, 30]

# Crea la figura y el eje del gráfico de barras
fig, ax = plt.subplots()
bars = ax.bar(categorias, valores)


# Función para actualizar los datos en cada fotograma
def update(frame):
    # Genera nuevos valores aleatorios
    nuevos_valores = [random.randint(5, 40) for _ in range(len(categorias))]

    # Actualiza las alturas de las barras
    for bar, nuevo_valor in zip(bars, nuevos_valores):
        bar.set_height(nuevo_valor)

    return bars


# Crea la animación
ani = FuncAnimation(fig, update, frames=range(100), interval=100)

# Muestra el gráfico dinámico
plt.show()
