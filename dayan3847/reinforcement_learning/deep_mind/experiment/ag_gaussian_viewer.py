from datetime import datetime

import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_control import viewer
from dm_env import TimeStep
import pyautogui

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgentGaussian import QLearningAgentGaussian

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

action_count = 7

ag = QLearningAgentGaussian(
    env=env,
    action_count=action_count,
)

f_name: str | None = None
r: float | None = None
counter: int | None = None
h_reward: list[float] | None = None
h_actions: list[int] | None = None


def init_episode():
    global ag, env, h_reward, h_actions, counter, f_name, r
    ag.init_episode()
    f_name = datetime.now().strftime('%Y%m%d%H%M%S')
    counter = 0
    r = 0
    h_reward = []
    h_actions = []


def action_to_mouse_position() -> np.array:
    global action_count

    _min = 765
    _max = 1795
    x, y = pyautogui.position()
    # print(f'La posición del cursor del mouse es: ({x}, {y})')
    x = np.clip(x, _min, _max)
    a = (x - _min) / (_max - _min)
    a *= (action_count - 1)
    a = int(round(a))

    return a


def policy_agent(time_step: TimeStep):
    global ag, env, h_reward, h_actions, counter, f_name, r
    if time_step.first():
        init_episode()
    else:
        r = time_step.reward
        h_reward.append(r)

    counter += 1

    a = action_to_mouse_position()
    h_actions.append(a)
    av = ag.action_values[a]
    print('action: {}({}) step: {}/{} r: {}'.format(a, av, counter, 1000, r))

    if counter == 1000:
        print('saving')
        np.savetxt(f'epc/{f_name}_actions.txt', h_actions)
        np.savetxt(f'epc/{f_name}_reward.txt', h_reward)
        np.savetxt('reward.txt', h_reward)

    return av


if __name__ == '__main__':
    viewer.launch(env, policy=policy_agent)
