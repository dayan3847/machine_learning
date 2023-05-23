import matplotlib.pyplot as plt
from dayan3847.bandit.src.BanditExperiment import BanditExperiment

if __name__ == '__main__':
    bandit_experiment1 = BanditExperiment(
        experiment_count=100,
        iterations=1000,
        actions_count=10,
        q1=5,
        epsilon=0,
    )
    bandit_experiment2 = BanditExperiment(
        experiment_count=100,
        iterations=1000,
        actions_count=10,
        q1=0.0,
        epsilon=0.1,
    )

    # result 1
    r1 = bandit_experiment1.run()
    # result 2
    r2 = bandit_experiment2.run()

    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.plot(r1['rewards_mean'], label='q1=5, e=0', color='black')
    plt.plot(r2['rewards_mean'], label='q1=0, e=0.1', color='gray')
    plt.legend()
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Reward Accumulated')
    plt.plot(r1['rewards_accumulated_mean'], label='q1=5, e=0', color='black')
    plt.plot(r2['rewards_accumulated_mean'], label='q1=0, e=0.1', color='gray')
    plt.legend()
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Optimal Action')
    plt.plot(r1['optimal_actions_mean'], label='q1=5, e=0', color='black')
    plt.plot(r2['optimal_actions_mean'], label='q1=0, e=0.1', color='gray')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
