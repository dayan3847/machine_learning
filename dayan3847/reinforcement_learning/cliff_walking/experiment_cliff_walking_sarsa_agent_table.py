import numpy as np
import matplotlib.pyplot as plt

from dm_env import TimeStep

from dayan3847.reinforcement_learning.cliff_walking.CliffWalkingEnvironment import CliffWalkingEnvironment, \
    ACTIONS_COUNT, BOARD_SHAPE
from dayan3847.reinforcement_learning.cliff_walking.CliffWalkingSarsaAgentTable import \
    CliffWalkingSarsaAgentTable

np.random.seed(0)

EXPERIMENTS: int = 200
EPISODES: int = 1000


def get_state(time_step: TimeStep) -> np.array:
    return time_step.observation['position']


if __name__ == '__main__':

    reward = np.zeros((EXPERIMENTS, EPISODES), dtype=np.float64)
    win = np.zeros(EPISODES, dtype=np.int64)

    for _ex in range(EXPERIMENTS):
        env = CliffWalkingEnvironment()
        ag = CliffWalkingSarsaAgentTable(
            actions_count=ACTIONS_COUNT,
            board_shape=BOARD_SHAPE,
        )


        def policy_agent(time_step: TimeStep) -> int:
            s = get_state(time_step)
            if not time_step.first():
                r = float(time_step.reward)
                ag.train_action(s, r)

            a = ag.select_an_action(s)

            return a


        for _ep in range(EPISODES):
            print('experiment: {} episode: {}'.format(_ex, _ep))

            _rew, _win = env.run_episode(policy=policy_agent)
            reward[_ex, _ep] = _rew
            if _win:
                win[_ep] += 1

    reward_mean = np.mean(reward, axis=0)

    plt.plot(reward_mean)
    plt.show()

    plt.plot(win)
    plt.show()

    np.savetxt('sarsa_reward.txt', reward_mean, delimiter=',')
    np.savetxt('sarsa_win.txt', win, delimiter=',')
