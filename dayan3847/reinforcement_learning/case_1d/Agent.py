import threading
import numpy as np
from dayan3847.reinforcement_learning.case_1d import Environment


class Agent:
    name: str
    env: Environment
    running: bool
    thread: threading.Thread
    point: np.array
    color: tuple[int, int, int]

    def __init__(self, env: Environment, name: str = 'Agent'):
        self.env = env
        self.name = name
        self.env.agents.append(self)
        self.running = False
        self.thread = threading.Thread(target=self.run_callback)
        self.color = (0, 0, 0)
        self.plot: bool = True
        self.init_point: np.array = np.array([0, 0])
        self.epoch: int = 0
        self.point = self.init_point
        self.reward: int = 0

    def run(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def run_callback(self):
        self.stop()
