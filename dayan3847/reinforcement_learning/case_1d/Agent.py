import threading
from abc import ABC

from dayan3847.reinforcement_learning.case_1d import Environment


class Agent(ABC):
    name: str
    env: Environment
    running: bool

    def __init__(self, env: Environment, name: str = 'Agent'):
        self.env = env
        self.name = name
        self.env.agents.append(self)
        self.running = False

    def run(self):
        self.running = True
        threading.Thread(target=self.run_callback).start()

    def stop(self):
        self.running = False

    def run_callback(self):
        self.stop()
