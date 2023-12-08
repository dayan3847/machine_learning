import numpy as np
from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_env import TimeStep

env: Environment = suite.load(domain_name="cartpole", task_name="balance")
action_spec = env.action_spec()


# Define a uniform random policy.
def random_policy(time_step: TimeStep):
    # time_step
    a = np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)
    # # siempre 0.1
    # a = np.full(action_spec.shape, 0.1)

    return a


# Launch the viewer application.
viewer.launch(env)
# viewer.launch(env, policy=random_policy)
