from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE) # needed to random init tasks
from metaworld.policies.sawyer_sweep_v2_policy import SawyerSweepV2Policy
import numpy as np
import time
import random


task = "sweep-v2-goal-observable"
policy_oracle = SawyerSweepV2Policy()

task_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
env = task_env()


for _ in range(3):
    obs = env.reset()

    obj_low = (-0.1, 0.6, 0.02)
    obj_high = (0.1, 0.7, 0.02)
    goal_low = (.49, .6, 0.00)
    goal_high = (0.51, .7, 0.02)

    random_init_pos_goal = np.random.uniform(goal_low, goal_high, 3)
    random_init_pos_obj = np.random.uniform(obj_low, obj_high, 3)

    env._set_obj_xyz(np.array(random_init_pos_obj))
    env._target_pos = np.array(random_init_pos_goal)

    for _ in range(498):
        env.render()
        # env.step(env.action_space.sample())
        # env.step(np.array([0, -1, 0, 0, 0]))
        action_teacher = policy_oracle.get_action(obs)
        obs, _, _, _ = env.step(np.array(action_teacher))
        # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
        time.sleep(0.05)
env.close()