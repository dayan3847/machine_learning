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

from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import get_action_values, get_state
from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgentGaussian import QLearningAgentGaussian

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

app = viewer.application.Application(title='Q-Learning Agent Gaussian')
action_count = 7
action_values: np.array = get_action_values(env, action_count)

ag = QLearningAgentGaussian(action_count)


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
    global ag, action_values
    s = get_state(time_step)
    if not time_step.first():
        r = float(time_step.reward)
        ag.train_action(s, r)
    else:
        r = 1
    a = ag.select_an_action(s)
    av = action_values[a]

    if r < .35:
        app._restart_runtime()

    return av


if __name__ == '__main__':
    # viewer.launch(env, policy=policy_agent)
    app.launch(env, policy=policy_agent)
