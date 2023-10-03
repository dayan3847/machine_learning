import matplotlib.pyplot as plt
import numpy as np

# Crear una matriz de ejemplo (reemplaza esto con tus datos)
matriz = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Trazar la matriz como una imagen
plt.imshow(matriz, cmap='viridis', interpolation='nearest')

# Añadir una barra de colores para indicar los valores
plt.colorbar()

# Mostrar el gráfico
plt.show()
