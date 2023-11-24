import pygame

# Inicializa Pygame
pygame.init()

# Inicializa el módulo joystick
pygame.joystick.init()

# Obtiene el número de joysticks
joystick_count = pygame.joystick.get_count()

# Si hay al menos un joystick
if joystick_count > 0:
    # Obtiene el primer joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Bucle principal
    while True:
        # Procesa los eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Obtiene los valores de la palanca
        axis_x = joystick.get_axis(0)  # Cambia el índice según la palanca que quieras leer
        axis_y = joystick.get_axis(1)  # Cambia el índice según la palanca que quieras leer

        # print(f"Valor de la palanca en X: {axis_x}, en Y: {axis_y}")
        print(f"Valor de la palanca en X: {axis_x}")
