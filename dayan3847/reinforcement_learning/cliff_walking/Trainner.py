import time
import numpy as np
import threading
from Agent import Environment, AgentQLearning


class Trainner:
    def __init__(
            self,
            agent: AgentQLearning,
            experiments_count: int = 50,
            episodes_count: int = 500,
    ):
        self.sleep: int = 0
        self.agent: AgentQLearning = agent
        self.env: Environment = agent.env
        self.experiments_status: tuple[int, int] = 0, experiments_count  # Experiment 0 of 50
        self.episodes_status: tuple[int, int] = 0, episodes_count  # Episode 0 of 500
        # rewards promedio(de todos los experimentos) por episodio
        self.rewards_sum: np.array = np.zeros(self.episodes_status[1])
        # cantidad de veces que llego al final
        self.success: np.array = np.zeros(self.episodes_status[1])
        self.thread: threading.Thread = threading.Thread(target=self.train_callback)

    def get_title(self):
        return 'Experiment: {}/{} Episode: {}/{}'.format(
            self.experiments_status[0] + 1,
            self.experiments_status[1],
            self.episodes_status[0] + 1,
            self.episodes_status[1]
        )

    def get_rewards_average(self) -> np.array:
        return self.rewards_sum / (self.experiments_status[0] + 1)

    def get_status(self) -> dict:
        q_best = np.ndarray((4, 12), dtype=int)
        for i in range(4):
            for j in range(12):
                state = np.array([i, j])
                actions: np.array = self.env.get_actions_available(state)
                a = self.agent.decide_an_action_best_q(actions, state)
                q_best[i, j] = a[0]

        return {
            'title': self.get_title(),
            'success': self.success.tolist(),
            'rewards': self.get_rewards_average().tolist(),
            'q_best': q_best.tolist(),
            'board_incidence': self.env.board_incidence.tolist(),
        }

    def train_callback(self):
        for i_exp in range(self.experiments_status[1]):
            self.experiments_status = i_exp, self.experiments_status[1]
            self.agent.reset_knowledge()
            for i_epi in range(self.episodes_status[1]):
                self.episodes_status = i_epi, self.episodes_status[1]
                print(self.get_title())
                episode_end: bool = False
                while not episode_end:
                    if self.sleep > 0:
                        time.sleep(self.sleep)
                    reward, episode_end = self.agent.run_step()
                    self.rewards_sum[i_epi] += reward
                    if episode_end and reward > 0:
                        self.success[i_epi] += 1

    def train(self):
        self.thread.start()
