import pygame
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import random
import time

# Inicializar pygame
pygame.init()

# Configuración de la ventana de pygame_gui
window_size = (800, 400)
window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Ventana con Gráfico")

# Configuración del gráfico de matplotlib
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvas(fig)

# Variables para el gráfico
values = [random.randint(1, 10) for _ in range(5)]
categories = [f"Category {i}" for i in range(1, 6)]
ax.bar(categories, values)
ax.set_title("Gráfico de Barras")

# Configurar pygame_gui
gui_manager = pygame_gui.UIManager(window_size)

# Crear un elemento de interfaz de usuario para mostrar la imagen del gráfico
graph_image = pygame_gui.elements.UIImage(
    relative_rect=pygame.Rect((50, 50), (300, 200)),
    image_surface=canvas.get_renderer().tostring_rgb(),
    manager=gui_manager,
)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Actualizar los datos del gráfico cada segundo
    if pygame.time.get_ticks() % 1000 == 0:
        values = [random.randint(1, 10) for _ in range(5)]
        ax.clear()
        ax.bar(categories, values)
        ax.set_title("Gráfico de Barras")
        canvas.draw()
        graph_image.set_image(canvas.get_renderer().tostring_rgb())

    gui_manager.process_events(event)
    gui_manager.update(1 / 60.0)

    window.fill((255, 255, 255))
    gui_manager.draw_ui(window)

    pygame.display.flip()

    clock.tick(60)

# Salir de pygame
pygame.quit()
