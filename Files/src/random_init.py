from metaworld.policies.sawyer_plate_slide_v2_policy import SawyerPlateSlideV2Policy
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
import numpy as np





for i_episode in range(20):
    plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-v2-goal-observable"]
    env = plate_slide_goal_observable_cls()
    observation = env.reset() # AL final de cada episodio se resetea
    print('env.reset')
    for t in range(100):
        if t == 0:
            print('Episode: ', i_episode, 'end-effector position: ', observation[:3],  'target position: ', observation[-3:], 'object (ball) position: ', observation[4:7])

        #env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
