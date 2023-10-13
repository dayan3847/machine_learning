import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from dayan3847.reinforcement_learning.case_1d import Agent


class Environment:
    MAX: np.array = np.array([12, 4])
    # TIME_STEP: float = .001
    TIME_STEP: float = .3

    def __init__(self, board_size: tuple[int, int]):
        self.board_size: tuple[int, int] = board_size
        self.agents: list[Agent] = []
        # Plot
        self.fig, self.ax = plt.subplots(nrows=4, ncols=5, figsize=(20, 5))
        self.fig.suptitle('Q-Learning')
        self.a_ql: Agent | None = None
        self.count_statuses: int = 0
        self.count_actions: int = 0
        self.q_actions_bars = None
        self.q_statuses_bars = None
        self.targets: list[Agent] = []

    def run(self):
        for agent in self.agents:
            agent.run()
        plotted: bool = self.plot()
        if plotted:
            self.stop()

    def stop(self):
        for agent in self.agents:
            agent.stop()

    def get_actions_available(self, ag: Agent) -> np.ndarray:
        actions: np.ndarray = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
        ])
        # if ag.point[0] == (-1 * self.MAX[0] + 1):
        #     actions = actions[1:]
        # elif ag.point[0] == (self.MAX[0] - 1):
        #     actions = actions[:-1]

        return actions

    def apply_action(self, ag: Agent, action: np.array) -> int:
        ag.point = ag.point + action
        for target in self.targets:
            if np.all(target.point == ag.point):
                ag.point = ag.init_point
                ag.epoch += 1
                return target.reward
        ag.point[0] = np.clip(ag.point[0], 0, self.MAX[0] - 1)
        ag.point[1] = np.clip(ag.point[1], 0, self.MAX[1] - 1)
        return -1

    def get_agent_by_name(self, name: str):
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def plot(self) -> bool:
        return False
