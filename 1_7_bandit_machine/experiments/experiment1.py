import numpy as np
import matplotlib.pyplot as plt

from dayan3847.bandit.src.BanditMachine import BanditMachine, BanditMachineAction
from dayan3847.bandit.src.BanditMachinePlayer import BanditMachinePlayerEpsilonGreedy

if __name__ == '__main__':
    print('experiment1')

    banditMachine: BanditMachine = BanditMachine(10)
    optimal_action_id: int = banditMachine.get_optimal_action()
    print('optimal_action: ', optimal_action_id)
    optimal_action: BanditMachineAction = banditMachine.action_list[optimal_action_id]
    print('optimal_action.median: ', optimal_action.median)

    # plot the bandit machine
    x = range(banditMachine.n_actions)
    y = [action.median for action in banditMachine.action_list]
    y = np.array(y)
    plt.xlabel('Action')
    plt.ylabel('Reward Mean')
    plt.scatter(x, y, color='blue', marker='h', s=500)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for xi in x:
        plt.axvline(x=xi, color='black', linestyle='-', alpha=0.5)
    plt.show()

    banditMachinePlayerEpsilonGreedy = BanditMachinePlayerEpsilonGreedy(banditMachine, epsilon=0, q1=5)
    iterations = range(1000)
    rewards = [0]
    rewards_accumulated = [0]
    optimal_actions = [0]
    q_bests = [banditMachinePlayerEpsilonGreedy.best_q]
    for i in iterations[1:]:
        reward, a = banditMachinePlayerEpsilonGreedy.play()
        rewards.append(reward)
        rewards_accumulated.append(rewards_accumulated[i - 1] + reward)
        optimal_actions.append(1 if (a == optimal_action_id) else 0)
        q_bests.append(banditMachinePlayerEpsilonGreedy.best_q)

    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.plot(iterations, rewards, label='epsilon greedy')
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Best Q')
    plt.plot(iterations, q_bests, label='best q')
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Reward Accumulated')
    plt.plot(iterations, rewards_accumulated, label='epsilon greedy')
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Optimal Action')
    plt.scatter(iterations, optimal_actions, label='epsilon greedy')
    plt.ylim(0, 1)
    plt.show()

    # print count optimal action equals to 1
    print('optimal_actions: ', sum(optimal_actions))
