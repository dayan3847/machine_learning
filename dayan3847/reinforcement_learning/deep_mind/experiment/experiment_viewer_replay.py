import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_control import viewer
from dm_env import TimeStep

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgentGaussian import QLearningAgentGaussian

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

f_name: str = '20231123221923'
app = viewer.application.Application(title='Replay "{}"'.format(f_name))

action_count = 7

ag = QLearningAgentGaussian(
    env=env,
    action_count=action_count,
)

counter: int = 0
h_actions = np.loadtxt(f'epc/{f_name}_actions.txt').astype(np.int32)
r = None


def init_episode():
    global ag, env, counter, r
    ag.init_episode()
    counter = 0
    r = 1


def policy_agent(time_step: TimeStep):
    global ag, env, h_actions, counter, f_name, r
    if time_step.first():
        init_episode()
    else:
        r = float(time_step.reward)

    a = h_actions[counter]
    counter += 1

    av = ag.action_values[a]
    print('action: {}({}) step: {}/{} r: {}'.format(a, av, counter, 1000, r))

    return av


if __name__ == '__main__':
    # viewer.launch(env, policy=policy_agent)
    app.launch(env, policy=policy_agent)
