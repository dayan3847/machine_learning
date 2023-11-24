import numpy as np
from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_env import TimeStep
import matplotlib.pyplot as plt
import pyautogui

# print actions

random_state = np.random.RandomState(42)
env: Environment = suite.load(domain_name='cartpole',
                              task_name='balance',
                              task_kwargs={
                                  'random': random_state,
                              },
                              visualize_reward=True)

h_reward = []
h_actions = []


def my_policy(time_step: TimeStep):
    global h_reward
    a = 0.1
    h_reward.append(time_step.reward)
    h_actions.append(a)

    # Obtén la posición del cursor del mouse
    x, _ = pyautogui.position()

    min = 1147
    max = 1413
    x = np.clip(x, min, max)

    a = (x - min) / (max - min)
    a *= 5
    a = int(round(a))


    # redondear

    print(f'La posición del cursor del mouse es: ({x}, _)')

    return np.array([a], dtype=env.action_spec().dtype)


# Launch the viewer application.
viewer.launch(env, my_policy)

# plot
plt.plot(h_reward)
plt.show()
plt.plot(h_actions)
plt.show()

# save in file txt
# np.savetxt('h_reward.txt', h_reward)
# np.savetxt('h_actions.txt', h_actions)
