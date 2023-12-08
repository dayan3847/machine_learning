import numpy as np

from dm_env import TimeStep

from dayan3847.reinforcement_learning.cliff_walking.CliffWalkingEnvironment import CliffWalkingEnvironment
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgent import TemporalDifferenceLearningAgent


def get_state(time_step: TimeStep) -> np.array:
    return time_step.observation['position']


def run_experiment(ag: TemporalDifferenceLearningAgent, experiments: int, episodes: int):
    reward = np.zeros((experiments, episodes), dtype=np.float64)
    win = np.zeros(episodes, dtype=np.int64)
    env = CliffWalkingEnvironment()

    for _ex in range(experiments):
        ag.knowledge_model.reset_knowledge()

        def policy_agent(time_step: TimeStep) -> int:
            s = get_state(time_step)
            if not time_step.first():
                r = float(time_step.reward)
                ag.train_action(s, r)

            a = ag.select_an_action(s)

            return a

        for _ep in range(episodes):
            _rew, _win = env.run_episode(policy=policy_agent)
            reward[_ex, _ep] = _rew
            if _win:
                win[_ep] += 1

            print(('\033[92m' if _win else '\033[91m') + 'experiment: {} episode: {}'.format(_ex, _ep) + '\033[0m')
        #     print(('\033[92m' if _win else '\033[91m') + '\u25A0' + '\033[0m', end='')
        # print()

    reward_mean = np.mean(reward, axis=0)

    np.savetxt(f'{ag.algorithm_name}_reward.txt', reward_mean, delimiter=',')
    np.savetxt(f'{ag.algorithm_name}_win.txt', win, delimiter=',')
