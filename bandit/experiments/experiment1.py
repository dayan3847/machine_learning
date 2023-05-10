import numpy as np

from bandit.entity.BanditMachine import BanditMachine, BanditMachineAction
import matplotlib.pyplot as plt

from bandit.entity.BanditMachinePlayer import BanditMachinePlayerEpsilonGreedy

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
    # scatter rectangulos verticales
    plt.scatter(x, y, color='blue', marker='h', s=500)

    # linea discontinua en y = 0, opacidad 0.5
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for xi in x:
        plt.axvline(x=xi, color='black', linestyle='-', alpha=0.5)
    plt.show()

    banditMachinePlayerEpsilonGreedy = BanditMachinePlayerEpsilonGreedy(banditMachine)
    iterations = range(1000)
    reward_i = 0
    rewards = [reward_i]
    q_bests = [banditMachinePlayerEpsilonGreedy.best_q]
    for i in iterations[1:]:
        reward_i += banditMachinePlayerEpsilonGreedy.play()
        rewards.append(reward_i)
        q_bests.append(banditMachinePlayerEpsilonGreedy.best_q)

    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    # grafico de linea
    plt.plot(iterations, rewards, label='epsilon greedy')
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Best Q')
    # grafico de linea
    plt.plot(iterations, q_bests, label='best q')
    plt.show()
