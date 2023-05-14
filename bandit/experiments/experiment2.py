import matplotlib.pyplot as plt
from bandit.entity.BanditExperiment import BanditExperiment

if __name__ == '__main__':
    bandit_experiment1 = BanditExperiment(
        experiment_count=100,
        iterations=1000,
        actions_count=10,
        q1=0.0,
        epsilon=0.1,
    )
    bandit_experiment2 = BanditExperiment(
        experiment_count=100,
        iterations=1000,
        actions_count=10,
        q1=0.0,
        epsilon=0.01,
    )
    bandit_experiment3 = BanditExperiment(
        experiment_count=100,
        iterations=1000,
        actions_count=10,
        q1=0.0,
        epsilon=0,
    )
    # bandit_experiment4 = BanditExperiment(
    #     experiment_count=100,
    #     iterations=1000,
    #     actions_count=10,
    #     q1=0.0,
    #     epsilon=0.2,
    # )


    # result 1
    r1 = bandit_experiment1.run()
    # result 2
    r2 = bandit_experiment2.run()
    # result 3
    r3 = bandit_experiment3.run()
    # result 4
    # r4 = bandit_experiment4.run()

    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.plot(r1['rewards_mean'], label='q1=0, e=0.1', color='black')
    plt.plot(r2['rewards_mean'], label='q1=0, e=0.01', color='red')
    plt.plot(r3['rewards_mean'], label='q1=0, e=0', color='green')
    # plt.plot(r4['rewards_mean'], label='q1=0, e=0.2', color='blue')
    plt.legend()
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Reward Accumulated')
    plt.plot(r1['rewards_accumulated_mean'], label='q1=0, e=0.1', color='black')
    plt.plot(r2['rewards_accumulated_mean'], label='q1=0, e=0.01', color='red')
    plt.plot(r3['rewards_accumulated_mean'], label='q1=0, e=0', color='green')
    # plt.plot(r4['rewards_accumulated_mean'], label='q1=0, e=0.2', color='blue')
    plt.legend()
    plt.show()

    plt.xlabel('Iterations')
    plt.ylabel('Optimal Action')
    plt.plot(r1['optimal_actions_mean'], label='q1=0, e=0.1', color='black')
    plt.plot(r2['optimal_actions_mean'], label='q1=0, e=0.01', color='red')
    plt.plot(r3['optimal_actions_mean'], label='q1=0, e=0', color='green')
    # plt.plot(r4['optimal_actions_mean'], label='q1=0, e=0.2', color='blue')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
