import matplotlib.pyplot as plt
import numpy as np

# Crear una matriz de colores (por ejemplo, 3x3)
color_matrix = np.array([[(128, 128, 128), (128, 128, 128), (128, 128, 128)]])

# Crear una matriz de bordes negros con las mismas dimensiones
border_matrix = np.zeros_like(color_matrix)

# Establecer los bordes en blanco (por ejemplo, valor 1.0)
border_matrix[1:-1, :] = 1.0
border_matrix[:, 1:-1] = 1.0

# Crear una figura y un eje para mostrar la matriz de colores
fig, ax = plt.subplots()

# Visualizar la matriz de colores con bordes negros
ax.imshow(color_matrix, cmap='viridis')
ax.imshow(border_matrix, cmap='gray', alpha=0.5)  # Usar un color claro para los bordes

# Ajustar los límites del eje
ax.set_xlim(-0.5, color_matrix.shape[1] - 0.5)
ax.set_ylim(-0.5, color_matrix.shape[0] - 0.5)

# Mostrar la trama
plt.show()
