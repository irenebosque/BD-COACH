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
from datetime import date



from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE) # needed to random init tasks
import math
from main_init import neural_network, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, render, max_time_steps_episode, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta, task, max_time_steps_per_repetition, \
number_of_repetitions, evaluation, save_results, env, task_short, dim_a, mountaincar_env, pendulum_env, metaworld_env, human_teacher, oracle_teacher, action_factor, kuka_env, cartpole_env, task_with_gripper, cut_feedback, evaluations_per_training, absolute_positions


print('evaluation: ', evaluation)
if oracle_teacher:
    from main_init import policy_oracle



e = agent.e
action_limit = agent.action_limit
buffer_size_max = agent.buffer_max_size
lr = agent.policy_model_learning_rate
HM_lr = agent.human_model_learning_rate
agent_with_hm_learning_rate = agent.agent_with_hm_learning_rate
buffer_sampling_size = agent.buffer_sampling_size





def oracle_gimme_feedback(action_teacher, action, h):


    difference = action_teacher - action
    #print("difference: ", difference)


    difference = np.array(difference)

    randomNumber = random.random()
    # if t_total < cut_feedback:
    #
    #     P_h = alpha * math.exp(-1 * tau * t_total)
    #
    # else:
    #     P_h = 0.0

    P_h = alpha * math.exp(-1 * tau * t_total)

    #hR = np.array([0, 0, 0]) #Rodrigo
    if randomNumber < P_h:


       # hR = np.sign(difference)#Rodrigo

        h = [0] * dim_a
        for i, name in enumerate(h):

            if abs(difference[i]) > theta:
                h[i] = np.sign(difference[i])

    #print("h: ", h)


    return h #TODO

def scale_observation(x, low, high):

    y = (((1 - (-1))/(high - low)) * (x - low)) - 1

    y = np.clip(y, -1, 1)
    return y

def scale_action_gripper(x, low, high):

    y = (((1 - (-1))/(high - low)) * (x - low)) - 1

    y = np.clip(y, low, high)
    return y

def scale_action_teacher_gripper(x):
    y = (-0.6 + 2*x) / 0.6


    #y = (0.6 +0.6*x)/2

    #y = np.clip(y, 0, 0.6)


    return y

def process_observation(observation):



    # What is the useful part of the observation
    if task == "drawer-open-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[4:6], observation[-3], observation[-2]))
    elif task == "button-press-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[3], observation[4:7]))
    elif task == "button-press-topdown-v2-goal-observable":
        #observation = np.hstack((observation[:3], observation[4:6], observation[-3], observation[-2]))
        observation = np.hstack(
            (observation[4:7] - observation[:3], observation[-3:] - observation[4:7]))
    elif task == "reach-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[-3:] ))
    elif task == "plate-slide-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[4:6], observation[-3], observation[-2]))
    elif task == "push-v2-goal-observable":
        #observation = np.hstack((observation[:3], observation[3],observation[4:7], observation[-3:]))
        observation = np.hstack((observation[:3], observation[3], observation[4:7], observation[-3], observation[-2]))
    elif task == "door-open-v2-goal-observable":
        # observation = np.hstack((observation[:3], observation[4:7], observation[-3:]))
       # observation = np.hstack((observation[4:7] - observation[:3], observation[-3:] - observation[4:7]))
        observation = np.hstack((observation[:3], observation[4:7], observation[-3:]))

    elif task == "assembly-v2-goal-observable":
        observation = np.hstack(
            (observation[4:7] - observation[:3], observation[-3:] - observation[4:7], observation[3]))
    elif task == "basketball-v2-goal-observable":
        # observation = np.hstack(
        #     (observation[:3], observation[3], observation[4:7],observation[-3:]))
        observation = np.hstack((observation[:3], observation[3], observation[4:7] , observation[-3], observation[-2]))
    elif task == "soccer-v2-goal-observable":
        #observation = np.hstack((observation[4:7] - observation[:3], observation[-3:] - observation[4:7]))
        #observation = np.hstack(( observation[:3],  observation[4:7], observation[-3:]))
        observation = np.hstack((observation[:3], observation[4:6], observation[-3], observation[-2]))
    elif task == "shelf-place-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[3], observation[4:7],observation[-3:]))
        observation = np.hstack((observation[4:7] - observation[:3], observation[3], observation[-3:] - observation[4:7]))
    elif task == "pick-place-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[3], observation[4:7],observation[-3:]))
        observation = np.hstack((observation[4:7] - observation[:3], observation[3], observation[-3:] - observation[4:7]))

    elif task == "sweep-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[4:7], observation[-3], observation[-2]))

    elif task == "button-press-topdown-wall-v2-goal-observable":
        observation = np.hstack((observation[:3], observation[4:6], observation[-3], observation[-2]))




    # low_env_boundary_ee  = low_env_boundary[:3]
    # high_env_boundary_ee = high_env_boundary[:3]
    #
    # low_env_boundary_gripper = low_env_boundary[3]
    # high_env_boundary_gripper = high_env_boundary[3]
    #
    # low_env_boundary_table = [-0.65, 0.25, 0.03]
    # high_env_boundary_table = [0.65, 0.95, 0.3]
    #
    # low_env_boundary_goal = low_env_boundary[-3:]
    # high_env_boundary_goal = high_env_boundary[-3:]
    #
    # observation[0] = scale_observation(observation[0], low_env_boundary_ee[0], high_env_boundary_ee[0])
    # observation[1] = scale_observation(observation[1], low_env_boundary_ee[1], high_env_boundary_ee[1])
    # observation[2] = scale_observation(observation[2], low_env_boundary_ee[2], high_env_boundary_ee[2])
    #
    #
    # #observation[3] = scale_observation(observation[3], low_env_boundary_gripper, high_env_boundary_gripper)
    #
    #
    # observation[3] = scale_observation(observation[3], low_env_boundary_table[0], high_env_boundary_table[0])
    # observation[4] = scale_observation(observation[4], low_env_boundary_table[1], high_env_boundary_table[1])
    # observation[5] = scale_observation(observation[5], low_env_boundary_table[2], high_env_boundary_table[2])
    #
    # #observation[-3] = scale_observation(observation[-3], low_env_boundary_goal[0], high_env_boundary_goal[0])
    # observation[-2] = scale_observation(observation[-2], low_env_boundary_goal[0], high_env_boundary_goal[0])
    # observation[-1] = scale_observation(observation[-1], low_env_boundary_goal[1], high_env_boundary_goal[1])
    # #
    #


    observation = [observation]
    # print('observation: ', observation)
    observation1 = tf.convert_to_tensor(observation, dtype=tf.float32)

    return observation1

def is_env_done(info):
    if info['success'] == 0:
        environment_done = False
        success_this_episode = 0

    else:

        success_this_episode = 1
        environment_done = True

    return environment_done, success_this_episode


def random_init_pos():
    # #soccer
    # goal_low = (-0.1, 0.8, 0.0)
    # goal_high = (0.1, 0.9, 0.0)
    # obj_low = (-0.1, 0.6, 0.03)
    # obj_high = (0.1, 0.7, 0.03)
    #
    # #Sweep
    #
    # obj_low = (-0.1, 0.6, 0.02)
    # obj_high = (0.1, 0.7, 0.02)
    # goal_low = (.49, .6, 0.00)
    # goal_high = (0.51, .7, 0.02)
    #
    # # Basket
    # obj_low = (-0.1, 0.6, 0.0299)
    # obj_high = (-0.1, 0.6, 0.0301)

    obj_low = (-0.1, 0.8, 0.115)  # -0.1 irene
    obj_high = (0.1, 0.9, 0.115)  # 0.1 irene


    goal_low = (-0.1, 0.85, 0.)
    goal_high = (0.1, 0.9, 0.)

    #random_init_pos_goal = np.random.uniform(goal_low, goal_high, 3)
    random_init_pos_obj = np.random.uniform(obj_low, obj_high, 3)
    #print("random_init_pos_obj", random_init_pos_obj)
    #self.obj_init_pos = goal_pos


    #print("random_init_pos_obj", random_init_pos_obj)
    #print("random_init_pos_goal", random_init_pos_goal)

    #env._set_obj_xyz(random_init_pos_obj)
    env.sim.model.body_pos[env.model.body_name2id('box')] = random_init_pos_obj
    #print("env.obj_init_pos", env.obj_init_pos)
    env._target_pos = env._get_site_pos('hole')
    # env._obj_to_target_init = abs(
    #     env._target_pos[2] - env._get_site_pos('buttonStart')[2]
    # )

    #env._target_pos = env.obj_init_pos + np.array([.0, -.16 - env.maxDist, .09])

    #env._target_pos = np.array(random_init_pos_goal)


    #env.sim.model.body_pos[env.model.body_name2id('puck_goal')] = env._target_pos

    #env._set_obj_xyz(np.array([0, 0.55, 0.020]))
    #env._target_pos = np.array(random_init_pos_goal)

    # env._target_pos = env.sim.model.site_pos[env.model.site_name2id('goal')] + env.sim.model.body_pos[
    #     env.model.body_name2id('shelf')]

    # for j in range(10):
    #     env.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
    #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    #     env.do_simulation([-1, 1], env.frame_skip)



# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)




results_counter = 0

weigths_counter = 0





task_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
env = task_env()
low_env_boundary = env.observation_space.low
print("low_env_boundary ", low_env_boundary )

high_env_boundary = env.observation_space.high
print("high_env_boundary", high_env_boundary)

repetition_is_over = False





for i_repetition in range(number_of_repetitions):

    results_counter_list = []

    if kuka_env:
        if rospy.is_shutdown():
            print('shutdown')
            break


    # Initialize variables
    total_pct_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_episodes, total_task, total_policy_loss_agent, total_policy_loss_hm, total_t = [alpha], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included], [tau], [0], [0], [0], [task_short], [0], [0], [0]
    show_human_model_lr, show_policy_lr, show_policy_batch_lr, show_absolute_pos, show_buffer_sampling_size = [HM_lr], [lr], [agent_with_hm_learning_rate], [absolute_positions], [buffer_sampling_size]

    total_success_per_episode = [[]] * evaluations_per_training
    total_success_per_episode[0].append(0)
    print("total_success_per_episode", total_success_per_episode)
    total_success_per_episode = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    total_success_per_episode = [[0], [0], [0], [0], [0]]


    t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback, episode_counter = 1, 0, 0, 0, 0, 0, 0, 0
    human_done, random_agent, evaluation_started = False, False, False
    repetition_list = []
    previous_time, time_this_t , time_this_rep = 0, 0, 0


    init_time = time.time()

    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%  GENERALPARAMTERS OF THE TEST %%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    #print('\nRepetition number:', i_repetition )
    print('Learning algorithm: ', agent_type)
    print('Evaluation?: ', evaluation)
    print('max_time_steps_episode: ', max_time_steps_episode)
    print('e: ' + str(e) + ' and ' + 'buffer size: ' + str(buffer_size_max))
    print('Human model included?: ' + str(agent.human_model_included) + '\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')


    agent.createModels(neural_network)



    # Start training loop
    for i_episode in range(0, max_num_of_episodes):

        #env = task_env()
        observation = env.reset()

        random_init_pos()



        success_this_episode = 0
        episode_counter += 1





        if repetition_is_over == True:
            # today = date.today()
            # d4 = today.strftime("%b-%d-%Y")
            # policy_model_weights = [agent.policy_model.get_weights()]
            # policy_model_weights = np.array(policy_model_weights)
            # human_model_weights = [agent.Human_model.get_weights()]
            # human_model_weights = np.array(human_model_weights)
            #
            # np.save(
            #     './weights/HM_weights' + 'HM-' + str(agent.human_model_included) + \
            #     '_e-' + str(e) + \
            #     '_B-' + str(buffer_size_max) + \
            #     '_tau-' + str(tau) + '_lr-' + str(lr) + '_HMlr-' + str(HM_lr) + '_agent_batch_lr-' + str(
            #         agent_with_hm_learning_rate) + '_task-' + task_short + '_rep-randm-0_2m_org_obs-big-net-B-sampling20-' + d4 + '.npy',
            #     human_model_weights)
            # np.save(
            #     './weights/Policy_weights' + 'HM-' + str(agent.human_model_included) + \
            #     '_e-' + str(e) + \
            #     '_B-' + str(buffer_size_max) + \
            #     '_tau-' + str(tau) + '_lr-' + str(lr) + '_HMlr-' + str(
            #         HM_lr) + '_agent_batch_lr-' + str(
            #         agent_with_hm_learning_rate) + '_task-' + task_short + '_rep-randm-0_2m_org_obs-big-net-B-sampling20-' + d4 + '.npy',
            #     policy_model_weights)

            repetition_is_over = False

            break

        if i_episode == 0:
            overwriteFiles = False
        if i_episode != 0:
            overwriteFiles = True




        print('Rep:', i_repetition, ', Episode:', i_episode, ', Rep timesteps:', t_total, "Computation time rep: ", str(time_this_rep)[:-5], 'Amount of feedback:', cummulative_feedback)




        doneButton = False







        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(1, max_time_steps_episode+1):




            #env.render(mode='human')
            # time.sleep(0.05)

            h = None

            secs = time.time() - init_time
            time_this_rep = str(datetime.timedelta(seconds=secs))


            # Finish repetition if the maximum number of steps per repetition is reached
            if t_total == (max_time_steps_per_repetition):
                print('repetition is over!')
                repetition_is_over = True
                #break

            observation_original = observation


            observation_processed = process_observation(observation)


            # Get action from oracle
            action_teacher = policy_oracle.get_action(observation_original)
            #action_teacher[-1]  = scale_action_teacher_gripper(action_teacher[-1])
            #action_teacher = np.clip(action_teacher, 0, 0.6)

           # print("action teacher: ", action_teacher)

            if task_with_gripper == False:
                action_teacher =  action_teacher[:-1].copy()


            # Get action from the agent
            action = agent.action(observation_processed)
            action_to_net = action
            #print("action before ", action)
            #action[-1] = scale_obs_gripper(action[-1])
            #print("action after ", action)
            action_to_env = action
            #action_to_env[-1] = np.clip(action_to_env[-1], 0, 0.6)
          #  print("action env ",action_to_env)




            if task_with_gripper == False:

                action_to_env = np.append(action, [1])

           # action_to_env[-1] = np.clip(action_to_env[-1], -1, 0.6)

            # Act
            observation, reward, environment_done, info = env.step(action_to_env)


            # Get feedback h from the oracle teacher
            h = oracle_gimme_feedback(action_teacher, action, h)







            environment_done, success_this_episode = is_env_done(info)








            # Compute done
            done = environment_done or repetition_is_over or t == max_time_steps_episode or doneButton #or feedback_joystick.ask_for_done()





            if np.any(h):
                h_counter += 1


            # Update weights
            if not evaluation and done == False:

                if agent.human_model_included:




                    agent.TRAIN_Human_Model_included(action_to_net, h, observation_processed, t_total, done )
                else:
                    agent.TRAIN_Human_Model_NOT_included(action_to_net, t_total, done, i_episode, h, observation_processed)



            t_total += 1



            # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
            r += reward




            # End of episode

            if done and (i_episode % 5 != 0):


                cummulative_feedback = cummulative_feedback + h_counter

                break

            if done and (i_episode % 5 == 0):

                for i_ev in range(0, evaluations_per_training):

                    print("\n")
                    print("%%%%%%%%%%%%%%")
                    print('Evaluating', str(i_ev+1), 'of', str(evaluations_per_training), '...')

                    success_this_episode = 0

                    #env = task_env()
                    observation = env.reset()

                    random_init_pos()



                    #env._set_obj_xyz(np.array([-0.09, 0.8, 0.03]))

                    # Iterate over the episode
                    for t_ev in range(0, max_time_steps_episode):

                        #env.render(mode='human')


                        observation_processed = process_observation(observation)

                        # Get action from the agent
                        action = agent.action(observation_processed)
                        action_to_net = action



                        if task_with_gripper == False:

                            action_to_env = np.append(action, [1])

                        #action[-1] = scale_obs_gripper(action[-1])
                        action_to_env = action
                       # action_to_env[-1] = np.clip(action_to_env[-1], -1, 0.6)
                        #print("action env ", action_to_env)



                        # Act
                        observation, reward, environment_done, info = env.step(action_to_env)


                        environment_done, success_this_episode = is_env_done(info)


                        # Compute done
                        done_evaluation = environment_done or repetition_is_over or t_ev == (max_time_steps_episode-2) or doneButton  # or feedback_joystick.ask_for_done()

                        if done_evaluation:
                            #env.close()

                            if environment_done == True:
                                print('Evaluation was successful :D')
                                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                            else:
                                print('Evaluation was a fail :(')
                                print("%%%%%%%%%%%%%%%%%%%%%%%%")





                            print("\n")
                            print('%%% END OF EVALUATION EPISODE %%%')


                            # The values inside the i_ev are shared by all the evaluations
                            if i_ev == 0:

                                total_episodes.append(episode_counter)
                                total_time_steps.append(t_total)
                                #total_success_per_episode.append(success_this_episode)
                                total_cummulative_feedback.append(cummulative_feedback)
                                total_pct_feedback.append(h_counter / (t + 1e-6))
                                show_e.append(e)
                                show_buffer_size.append(buffer_size_max)
                                show_human_model.append(agent.human_model_included)
                                show_tau.append(tau)
                                show_human_model_lr.append(HM_lr)
                                show_policy_lr.append(lr)
                                show_policy_batch_lr.append(agent_with_hm_learning_rate)
                                show_absolute_pos.append(absolute_positions)
                                show_buffer_sampling_size.append(buffer_sampling_size)

                            total_success_per_episode[i_ev].append(success_this_episode)











                            if (save_results):

                                df = pd.DataFrame({'Episode': total_episodes,
                                                   'Timesteps': total_time_steps,
                                                   'Success': total_success_per_episode[i_ev],
                                                   'Feedback': total_cummulative_feedback,
                                                   'Percentage_feedback': total_pct_feedback,
                                                   'e': show_e,
                                                   'Buffer_size': show_buffer_size,
                                                   'Human_model': show_human_model,
                                                   'Tau': show_tau,
                                                   'Human_model_lr': show_human_model_lr,
                                                   'Policy_lr': show_policy_lr,
                                                   'Policy_batch_lr': show_policy_batch_lr,
                                                   'Absolute_pos': show_absolute_pos,
                                                   'Buffer_sampling_size': show_buffer_sampling_size})






                                path_results = './results/HM-' + str(agent.human_model_included) + \
                                                   '_e-' + str(e) + \
                                                   '_B-' + str(buffer_size_max)  + \
                                                   '_task-' + task_short  + \
                                                   '_absolute_pos-' + str(absolute_positions) + \
                                                   '_rep-alpha07v2-' + str(results_counter).zfill(2) + '.csv'


                                if overwriteFiles == False:
                                    while os.path.isfile(path_results):
                                        results_counter += 1
                                        path_results = './results/HM-' + str(agent.human_model_included) + \
                                                   '_e-' + str(e) + \
                                                   '_B-' + str(buffer_size_max)  + \
                                                   '_task-' + task_short  + \
                                                   '_absolute_pos-' + str(absolute_positions) + \
                                                   '_rep-alpha07v2-' + str(results_counter).zfill(2) + '.csv'

                                if i_episode == 0:
                                    results_counter_list.append(results_counter)
                                #print("counterr",results_counter_list)
                                #print('iev', i_ev)

                                print("Success rep ",results_counter_list[i_ev], ': ',total_success_per_episode[i_ev])
                                df.to_csv('./results/HM-' + str(agent.human_model_included) + \
                                                   '_e-' + str(e) + \
                                                   '_B-' + str(buffer_size_max)  + \
                                                   '_task-' + task_short  + \
                                                   '_absolute_pos-' + str(absolute_positions) + \
                                                   '_rep-alpha07v2-' + str(results_counter_list[i_ev]).zfill(2) + '.csv', index=False)



                                #env.close()



                            break

                break


