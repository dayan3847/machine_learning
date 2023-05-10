import numpy as np

from bandit.entity.BanditMachine import BanditMachine
from bandit.entity.BanditMachinePlayer import BanditMachinePlayerEpsilonGreedy


class BanditExperiment:
    def __init__(
            self,
            experiment_count: int = 100,
            iterations: int = 1000,
            actions_count: int = 10,
            q1: float = 0.0,
            epsilon: float = 0.1,
    ):
        self.experiment_count = experiment_count
        self.iterations = iterations
        self.actions_count = actions_count
        self.q1 = q1
        self.epsilon = epsilon

    def run(self):
        rewards = np.zeros((self.experiment_count, self.iterations))
        rewards_accumulated = np.zeros((self.experiment_count, self.iterations))
        optimal_actions = np.zeros((self.experiment_count, self.iterations))

        for i in range(self.experiment_count):
            bandit_machine: BanditMachine = BanditMachine(self.actions_count)
            optimal_action = bandit_machine.get_optimal_action()
            bandit_machine_player_epsilon_greedy = BanditMachinePlayerEpsilonGreedy(
                bandit_machine,
                q1=self.q1,
                epsilon=self.epsilon,
            )
            for j in range(self.iterations)[1:]:
                reward, a = bandit_machine_player_epsilon_greedy.play()
                rewards[i, j] = reward
                rewards_accumulated[i, j] = rewards_accumulated[i, j - 1] + reward
                optimal_actions[i, j] = 1 if a == optimal_action else 0

        rewards_mean = rewards.mean(axis=0)
        rewards_accumulated_mean = rewards_accumulated.mean(axis=0)
        optimal_actions_mean = optimal_actions.mean(axis=0)

        return {
            'rewards_mean': rewards_mean,
            'rewards_accumulated_mean': rewards_accumulated_mean,
            'optimal_actions_mean': optimal_actions_mean,
        }
