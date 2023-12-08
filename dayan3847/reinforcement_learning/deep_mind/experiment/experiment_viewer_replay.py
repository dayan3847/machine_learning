import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_control import viewer
from dm_env import TimeStep
import imageio

from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import get_action_values

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

f_name: str = '20231124021704'

app = viewer.application.Application(title='Replay "{}"'.format(f_name))
action_count = 7
action_values: np.array = get_action_values(env, action_count)

h_actions = np.loadtxt(f'epc/{f_name}_actions.txt').astype(np.int32)
counter: int = 0
r = None


def init_episode():
    global counter, r
    counter = 0
    r = 1


def policy_agent(time_step: TimeStep):
    global h_actions, counter, r
    if time_step.first():
        init_episode()
    else:
        r = float(time_step.reward) * 1e3

    a = h_actions[counter]
    counter += 1
    frame = env.physics.render(camera_id=0)
    # save
    imageio.imwrite(f'frames/f{counter}.png', frame)

    av = action_values[a]
    print('action12: {}({}) step: {}/{} r: {}'.format(a, av, counter, 1000, r))

    return av


if __name__ == '__main__':
    # viewer.launch(env, policy=policy_agent)
    app.launch(env, policy=policy_agent)
