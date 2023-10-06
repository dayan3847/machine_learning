import time
import numpy as np
from dayan3847.reinforcement_learning.case_1d import Agent


class AgentDynamic(Agent):

    def run_callback(self):
        while self.running:
            time.sleep(self.env.TIME_STEP)
            action: np.array = self.get_action()[0]
            self.env.apply_action(self, action)

    def get_action(self) -> (np.array, int):
        pass
