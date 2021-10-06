
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE) # needed to random init tasks
from metaworld.policies.sawyer_sweep_v2_policy import SawyerSweepV2Policy
import numpy as np
import time
import random
import datetime
from datetime import date
import math
from main_init import neural_network, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, render, max_time_steps_episode, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta, task, max_time_steps_per_repetition, \
number_of_repetitions, evaluation, save_results, env, task_short, dim_a, mountaincar_env, pendulum_env, \
    metaworld_env, human_teacher, oracle_teacher, action_factor, kuka_env, cartpole_env, task_with_gripper, \
    cut_feedback, evaluations_per_training, absolute_positions

e = agent.e
action_limit = agent.action_limit
buffer_size_max = agent.buffer_max_size
lr = agent.policy_model_learning_rate
HM_lr = agent.human_model_learning_rate
agent_with_hm_learning_rate = agent.agent_with_hm_learning_rate
buffer_sampling_size = agent.buffer_sampling_size

task = "sweep-v2-goal-observable"
policy_oracle = SawyerSweepV2Policy()

task_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
env = task_env()









def random_init_pos():

    goal_low = (-0.1, 0.8, 0.0)
    goal_high = (0.1, 0.9, 0.02)
    random_init_pos_goal = np.random.uniform(goal_low, goal_high, 3)

    obj_low = (-0.1, 0.5, 0.019)
    obj_high = (0.1, 0.6, 0.021)
    random_init_pos_obj = np.random.uniform(obj_low, obj_high, 3)

    env._set_obj_xyz(np.array(random_init_pos_obj))
    shelf_pos = np.array(random_init_pos_goal)
    #env.sim.model.body_pos[env.model.body_name2id('shelf')] = shelf_pos
    env._target_pos = shelf_pos

    # env._target_pos = env.sim.model.site_pos[env.model.site_name2id('goal')] + env.sim.model.body_pos[
    #     env.model.body_name2id('shelf')]

    for j in range(10):
        env.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
        env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        env.do_simulation([-1, 1], env.frame_skip)


def is_env_done(info):
    if info['success'] == 0:
        environment_done = False


    else:

        environment_done = True

    return environment_done

# MAIN LOOP
for i_rep in range(10):
    "\n"
    print('rep: ', i_rep)

    for i_episode in range(10):
        "\n"
        print('i_episode: ', i_episode)
        obs = env.reset()
        random_init_pos()


        for t in range(1, 501):

            env.render()
            action_teacher = policy_oracle.get_action(obs)


            obs, reward, environment_done, info = env.step(np.array(action_teacher))
            print()
            time.sleep(0.05)

            environment_done = is_env_done(info)
            if t == (500-2) or environment_done:
                env.close()

