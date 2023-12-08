from datetime import datetime

import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_control import viewer
from dm_env import TimeStep

from dayan3847.reinforcement_learning.deep_mind.experiment.agests import balance_example_5_11111
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import get_action_values, get_state

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

app = viewer.application.Application(title='Q-Learning Agent Gaussian')

ag = balance_example_5_11111()
# ag.knowledge_model.load_knowledge('knowledge.csv')

action_count = ag.action_count
action_values: np.array = get_action_values(env, action_count)

counter: int | None = None
f_name: str | None = None
r: float | None = None
h_reward: list[float] | None = None
h_actions: list[int] | None = None


def init_episode():
    global h_reward, h_actions, counter, f_name, r
    f_name = datetime.now().strftime('%Y%m%d%H%M%S')
    counter = 0
    r = 1
    h_reward = []
    h_actions = []


def policy_agent(time_step: TimeStep):
    global ag, action_values, h_actions, counter, r
    s = get_state(time_step)
    if time_step.first():
        init_episode()
    else:
        r = float(time_step.reward)
        ag.train_action(s, r)

    a = ag.select_an_action(s)
    counter += 1

    av = action_values[a]

    if r < .35:
        app._restart_runtime()

    if counter % 10 == 0:
        print('saving knowledge')
        ag.knowledge_model.save_knowledge('epc/{}_knowledge.csv'.format(f_name))
        ag.knowledge_model.save_knowledge('knowledge.csv')

    print('action: {}({}) step: {}/{} r: {}'.format(a, av, counter, 1000, r))
    return av


if __name__ == '__main__':
    # viewer.launch(env, policy=policy_agent)
    app.launch(env, policy=policy_agent)
