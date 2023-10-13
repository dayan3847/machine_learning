import numpy as np
import pygame
from dayan3847.reinforcement_learning.case_2d import Environment
from dayan3847.reinforcement_learning.case_1d import Agent


class AgentGrapherPyGame(Agent):
    R = 20

    def __init__(self, env: Environment):
        super().__init__(env)
        self.plot = False
        self.window = None
        self.window_size = (800, 600)

    def draw(self):
        # background color
        self.window.fill((255, 255, 255))
        # print grid
        for _x in range(self.env.MAX[0]):
            for _y in range(self.env.MAX[1]):
                p = self.convert_point((_x, _y)),
                pygame.draw.circle(
                    self.window,
                    (200, 200, 200),
                    self.convert_point((_x, _y)),
                    AgentGrapherPyGame.R,
                )
        for agent in self.env.agents:
            if not agent.plot:
                continue
            pygame.draw.circle(
                self.window,
                agent.color,
                self.convert_point(agent.point),
                AgentGrapherPyGame.R,
            )
        pygame.display.update()

    def convert_point(self, point):
        line_x = np.linspace(0, self.window_size[0], self.env.MAX[0] + 2)[1:-1]
        line_y = np.linspace(0, self.window_size[1], self.env.MAX[1] + 2)[1:-1]
        # inverse order
        line_y = line_y[::-1]
        return (
            int(line_x[point[0]]),
            int(line_y[point[1]]),
        )

    def run_callback(self):
        pygame.display.set_caption('Pointing Simulation')
        self.window = pygame.display.set_mode(self.window_size)
        while self.running:
            self.draw()
            self.check_close()
        pygame.quit()

    def check_close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.stop()
        keys = pygame.key.get_pressed()
        # if press 'q' key, stop simulation
        if keys[pygame.K_q]:
            self.env.stop()
