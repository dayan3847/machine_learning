import threading

import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment


class Agent:
    name: str
    env: Environment
    running: bool
    point: np.array
    color: tuple[int, int, int]

    def __init__(self, env: Environment, name: str = 'Agent'):
        self.env = env
        self.name = name
        self.env.agents.append(self)
        self.running = False
        self.point = np.array([0, 0])
        self.color = (0, 0, 0)

    def run(self):
        self.running = True
        threading.Thread(target=self.run_callback).start()

    def stop(self):
        self.running = False

    def run_callback(self):
        self.stop()
