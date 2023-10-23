import matplotlib.pyplot as plt

from Trainner import Trainner


class TrainnerPlotter:
    def __init__(self, trainner: Trainner):
        self.trainner: Trainner = trainner

    # Reward Average
    @staticmethod
    def plot_rewards_average(trainner: Trainner):
        rewards_average = trainner.get_rewards_average()
        plt.plot(rewards_average, label='Reward Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward Average')
        plt.show()

    # Success
    @staticmethod
    def plot_success(trainner: Trainner):
        success = trainner.success
        plt.plot(success, label='Success')
        plt.xlabel('Episode')
        plt.ylabel('Success')
        plt.show()

    # Best Action
    @staticmethod
    def plot_best_action(trainner: Trainner):
        best_actions = trainner.agent.get_best_actions_for_all_states()
        plt.matshow(best_actions, label='Best Action')
        m, n = best_actions.shape
        for i in range(m):
            for j in range(n):
                _cell = best_actions[i, j]
                cell = '↑' if _cell == 0 \
                    else '↓' if _cell == 1 \
                    else '←' if _cell == 2 \
                    else '→' if _cell == 3 \
                    else 'X'
                plt.text(j, i, cell, ha='center', va='center', color='red', fontsize=50)

        plt.xlabel('State_0')
        plt.ylabel('State_1')
        plt.show()

    # Board Incidence
    @staticmethod
    def plot_board_incidence(trainner: Trainner):
        plt.matshow(trainner.env.board_incidence, label='Board Incidence')
        m, n = trainner.env.board_incidence.shape
        for i in range(m):
            for j in range(n):
                plt.text(j, i, trainner.env.board_incidence[i, j], ha='center', va='center', fontsize=11)

        plt.xlabel('State_0')
        plt.ylabel('State_1')
        plt.colorbar()
        plt.set_cmap('plasma')
        plt.show()
