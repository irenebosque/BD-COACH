import numpy as np
import pandas as pd
import time
import datetime
import tensorflow as tf
import os
import random
from tabulate import tabulate
import rospy
import matplotlib.pyplot as plt



from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE) # needed to random init tasks
import math
from main_init import neural_network, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, render, max_time_steps_episode, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta, task, max_time_steps_per_repetition, \
number_of_repetitions, evaluation, save_results, env, task_short, dim_a, mountaincar_env, pendulum_env, metaworld_env, human_teacher, oracle_teacher, action_factor, kuka_env, h_threshold, cartpole_env
print('evaluation: ', evaluation)
if oracle_teacher:
    from main_init import policy_oracle



e = agent.e
action_limit = agent.action_limit
buffer_size_max = agent.buffer_max_size
lr = agent.policy_model_learning_rate
HM_lr = agent.human_model_learning_rate

if kuka_env:
    rospy.init_node('agent_control')



"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""



# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)




results_counter = 0
weigths_counter = 0





task_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
env = task_env()

repetition_is_over = False





for i_repetition in range(number_of_repetitions):
    if kuka_env:
        if rospy.is_shutdown():
            print('shutdown')
            break


    # Initialize variables
    total_pct_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_episodes, total_task, total_policy_loss_agent, total_policy_loss_hm, total_success_per_episode, total_t = [alpha], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included], [tau], [0], [0], [0], [task_short], [0], [0], [0], [0]
    t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback, success_counter, episode_counter = 1, 0, 0, 0, 0, 0, 0, 0, 0
    human_done, random_agent, evaluation_started = False, False, False
    repetition_list = []
    previous_time, time_this_t , time_this_rep = 0, 0, 0





    # Start training loop
    for i_episode in range(0, max_num_of_episodes):
        #env = task_env()
        observation = env.reset()


        success_this_episode = 0
        episode_counter += 1





        if repetition_is_over == True:
            repetition_is_over = False
            break

        if i_episode == 0:
            overwriteFiles = False
        if i_episode != 0:
            overwriteFiles = True



        print("\n")
        print('Rep:', i_repetition, ', Episode:', i_episode, ', Rep timesteps:', t_total, "Computation time rep: ", str(time_this_rep)[:-5], 'Amount of feedback:', cummulative_feedback)




        doneButton = False







        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(1, max_time_steps_episode+1):


            env.render(mode='human')






            # Finish repetition if the maximum number of steps per repetition is reached
            if t_total == (max_time_steps_per_repetition):
                print('repetition is over!')
                repetition_is_over = True
                #break


            action_teacher = policy_oracle.get_action(observation)

            print('action: ', action_teacher)



            # Act
            observation, reward, environment_done, info = env.step(action_teacher) #gripper






            #if metaworld_env:
            if info['success'] == 0:
                environment_done = False

            else:
                success_counter += 1

                success_this_episode = 1
                environment_done = True







            # Compute done
            done = environment_done or repetition_is_over or t == max_time_steps_episode or doneButton #or feedback_joystick.ask_for_done()


            # End of episode

            if done:


                break







