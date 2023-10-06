import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from dayan3847.reinforcement_learning.case_1d import Agent


class Environment:
    MAX: np.array = np.array([10, 1])
    # TIME_STEP: float = .001
    TIME_STEP: float = .03

    def __init__(self):
        self.agents: list[Agent] = []
        # Plot
        self.fig, self.ax = plt.subplots(nrows=2, ncols=10, figsize=(20, 5))
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
        actions: np.ndarray = np.array([np.array([_x, 0]) for _x in range(-1, 2)])
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
                return target.reward
        ag.point[0] = np.clip(ag.point[0], -1 * self.MAX[0] + 1, self.MAX[0] - 1)
        return -1

    def get_agent_by_name(self, name: str):
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def plot(self) -> bool:
        self.a_ql = self.get_agent_by_name('q_learning')
        if self.a_ql is None:
            return False
        self.count_statuses = self.a_ql.Q.shape[0]
        self.count_actions = self.a_ql.Q.shape[1]

        self.q_actions_bars: np.ndarray = np.empty((self.count_actions,), dtype=object)
        self.q_statuses_bars: np.ndarray = np.empty((self.count_statuses,), dtype=object)

        # statuses: np.array = np.arange(self.count_statuses)
        # for i in range(self.count_actions):
        #     values: np.array = self.abs(self.a_ql.Q[:, i])
        #     self.q_actions_bars[i] = self.ax[i].bar(statuses, values, label='Status')
        #     self.ax[i].set_ylim(0.8, 1)
        # self.ax[0].set_title('Left')
        # self.ax[1].set_title('No Move')
        # self.ax[2].set_title('Right')

        actions: np.array = np.arange(self.count_actions)
        for i in range(self.count_statuses):
            values: np.array = self.abs(self.a_ql.Q[i])
            self.q_statuses_bars[i] = self.ax[0][i].bar(actions, values, label='Action')
            self.ax[0][i].set_ylim(.32, .34)
            self.ax[0][i].set_title(f'Status {i}')

            self.ax[1][i].imshow(self.a_ql.Q[i].reshape(1, -1), cmap='viridis', interpolation='nearest')
            self.ax[1][i].set_title(f'Q Status {i}')
        # colorbar
        # self.fig.colorbar(self.ax[3].get_images()[0], ax=self.ax[3])

        # create animation using the animate() function
        _ani = animation.FuncAnimation(self.fig, self.plot_callback, interval=17)  # 60 fps

        plt.tight_layout()
        plt.show()

        return True

    def plot_callback(self, frame):
        # for i in range(self.count_actions):
        #     values: np.array = self.abs(self.a_ql.Q[:, i])
        #     for bar, v in zip(self.q_actions_bars[i], values):
        #         bar.set_height(v)

        for i in range(self.count_statuses):
            values: np.array = self.abs(self.a_ql.Q[i])
            for bar, v in zip(self.q_statuses_bars[i], values):
                bar.set_height(v)
            self.ax[1][i].imshow(self.a_ql.Q[i].reshape(1, -1), cmap='viridis', interpolation='nearest')

    @staticmethod
    def abs(values):
        values = np.abs(values)
        # values = values / np.max(values)
        values = values / np.sum(values)
        return values
