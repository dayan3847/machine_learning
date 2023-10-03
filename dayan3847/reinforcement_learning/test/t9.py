import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo para las barras
categorias = ['A', 'B', 'C', 'D', 'E']
valores = [10, 24, 15, 30, 18]

# Crear un gráfico de barras
plt.bar(categorias, valores)

# Agregar etiquetas (valores) en la parte superior de cada barra
for i, valor in enumerate(valores):
    plt.text(categorias[i], valor, str(valor), ha='center', va='bottom')

# Configuración adicional del gráfico
plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Gráfico de Barras con Etiquetas de Valores')

# Mostrar el gráfico
plt.show()
