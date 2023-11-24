from datetime import datetime

import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_control import viewer
from dm_env import TimeStep

# mouse
import pyautogui
# joystick
# import pygame

from dayan3847.reinforcement_learning.deep_mind.agent.Agent import get_action_values

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

app = viewer.application.Application()

action_count = 7

action_values: np.array = get_action_values(env, action_count)

f_name: str | None = None
r: float | None = None
counter: int | None = None
h_reward: list[float] | None = None
h_actions: list[int] | None = None


def init_episode():
    global env, h_reward, h_actions, counter, f_name, r
    f_name = datetime.now().strftime('%Y%m%d%H%M%S')
    counter = 0
    r = 1
    h_reward = []
    h_actions = []


def action_from(x, _min, _max) -> np.array:
    global action_count
    x = np.clip(x, _min, _max)
    a = (x - _min) / (_max - _min)
    a *= (action_count - 1)
    a = int(round(a))

    return a


def action_from_mouse() -> np.array:
    _min = 765
    _max = 1795
    x, y = pyautogui.position()
    # print(f'La posición del cursor del mouse es: ({x}, {y})')

    return action_from(x, _min, _max)


# pygame.init()
# pygame.joystick.init()
#
# joystick = None
# if pygame.joystick.get_count() > 0:
#     joystick = pygame.joystick.Joystick(0)
#     joystick.init()
# else:
#     raise Exception('no joystick found')


# def action_from_joystick() -> np.array:
#     global joystick
#     _min = -.69
#     _max = .78
#     x = joystick.get_axis(0)
#     # y = joystick.get_axis(1)
#     # print(f'La posición del joystick es: ({x}, {y})')
#
#     return action_from(x, _min, _max)


def policy_agent(time_step: TimeStep):
    global env, h_reward, h_actions, counter, f_name, r
    if time_step.first():
        init_episode()
    else:
        r = time_step.reward
        h_reward.append(r)

    counter += 1

    a = action_from_mouse()
    # a = action_from_joystick()
    h_actions.append(a)
    av = action_values[a]
    print('action: {}({}) step: {}/{} r: {}'.format(a, av, counter, 1000, r))

    if r < .35:
        app._restart_runtime()

    if counter == 1000:
        print('saving')
        np.savetxt(f'epc/{f_name}_actions.txt', h_actions)
        np.savetxt(f'epc/{f_name}_reward.txt', h_reward)
        np.savetxt('reward.txt', h_reward)

    return av


if __name__ == '__main__':
    # viewer.launch(env, policy=policy_agent)
    app.launch(env, policy=policy_agent)
