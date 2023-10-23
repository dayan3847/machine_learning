import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Crear una figura y un eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función para actualizar la superficie
def update_surface(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Superficie 3D Cambiante')

    # Generar datos aleatorios para la superficie
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)

    # Crear la superficie de puntos
    ax.scatter(x, y, z, c='b', marker='o')

# Crear la animación que llama a la función update_surface cada segundo
ani = FuncAnimation(fig, update_surface, interval=1000)

# Mostrar la animación
plt.show()

