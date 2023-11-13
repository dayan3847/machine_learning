from datetime import datetime
import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from QLearningAgentPV import QLearningAgentPV
from dm_env import StepType

if __name__ == '__main__':
    random_state = np.random.RandomState(42)
    env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

    ag: QLearningAgentPV = QLearningAgentPV(
        env=env,
        action_count=11,
    )
    ag.knowledge_model.load_knowledge('knowledge.h5')

    while True:
        if np.loadtxt('stop.txt') != 0:
            print('stop')
            break
        name: str = datetime.now().strftime('%Y%m%d%H%M%S')
        print(f'running episode {name}')
        ag.init_episode()
        history_frames: list[np.array] = [ag.get_current_frame()]
        while StepType.LAST != ag.time_step.step_type:
            if np.loadtxt('stop.txt') == 2:
                print('stop')
                break
            print('\033[92m{}\033[00m'.format(ag.step))
            _r, _a = ag.run_step()
            history_frames.append(ag.get_current_frame())
            print("Reward: ", _r)
            print("Action: ", _a)
        print('saving knowledge')
        ag.knowledge_model.save_knowledge('knowledge.h5')
        print('saving reward')
        np.savetxt('reward.txt', ag.history_reward)
        np.savetxt(f'ep/{name}_reward.txt', ag.history_reward)
        print('saving frames')
        np.save('frames.npy', history_frames)
