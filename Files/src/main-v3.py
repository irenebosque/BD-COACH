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
from load_weights import loadWeights

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






repetition_is_over = False





for i_repetition in range(number_of_repetitions):
    if kuka_env:
        if rospy.is_shutdown():
            print('shutdown')
            break


    # Initialize variables
    total_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_episodes, total_task, total_policy_loss_agent, total_policy_loss_hm, total_success_per_episode, total_t = [alpha], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included], [tau], [0], [0], [0], [task_short], [0], [0], [0], [0]
    t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback, success_counter, episode_counter = 1, 0, 0, 0, 0, 0, 0, 0, 0
    human_done, random_agent, evaluation_started = False, False, False
    repetition_list = []


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

        if i_episode % 5 == 0:
            print('episode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ', i_episode)

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
        print('Starting episode number: ', i_episode)

        doneButton = False

        plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-v2-goal-observable"]
        env = plate_slide_goal_observable_cls()
        observation = env.reset()




        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(1, max_time_steps_episode+1):

            #env.render(mode='human')



            h = None
            secs = time.time() - init_time
            time_plot = str(datetime.timedelta(seconds=secs))


            # Print info
            if t % 100 == 0:
                print("\n")
                print('Rep: ', i_repetition, 'Episode: ', i_episode, "t this ep: ", t, ' total timesteps: ', t_total, 'time: ', time_plot)

            # Finish repetition if the maximum number of steps per repetition is reached
            if t_total == (max_time_steps_per_repetition):
                print('repetition is over!')
                repetition_is_over = True
                #break

            if metaworld_env:
                observation_original = observation
                #print("observation_original", observation_original)
                # What is the useful part of the observation
                if task == "drawer-open-v2-goal-observable":
                    observation = np.hstack((observation[:3], observation[4:7]))
                elif task == "button-press-v2-goal-observable":
                    observation = np.hstack((observation[:3], observation[3], observation[4:7]))
                elif task == "button-press-topdown-v2-goal-observable":
                    observation = np.hstack((observation[:3], observation[3], observation[4:7]))
                elif task == "reach-v2-goal-observable":
                    observation = np.hstack((observation[:3], observation[3], observation[-3:]))
                elif task == "plate-slide-v2-goal-observable":
                    observation = np.hstack((observation[:3], observation[3], observation[4:7], observation[-3:]))



            # Prepare observation

            observation = [observation]
            observation1 = tf.convert_to_tensor(observation, dtype=tf.float32)

            if kuka_env:
                observation1 = tf.reshape( observation1, [1, agent.dim_o])
            else:
                observation1 = observation1


                # Get action from the agent
            action = agent.action(observation1)


            # For the pressing button topdown task:
            action_append_gripper = np.append(action, [1])






            # Act
            observation, reward, environment_done, info = env.step(action_append_gripper)



            if not evaluation:

                # Get heedback h from a real human or from an oracle
                if human_teacher:



                    for i, name in enumerate(h):
                        if abs(h[i]) < h_threshold:
                            h[i] = 0
                        else:
                            h[i] = h[i]




                elif oracle_teacher:

                    action_teacher = policy_oracle.get_action(observation_original)
                    action_teacher = np.clip(action_teacher, -1, 1)
                    #print("action_teacherl",action_teacher)
                    action_teacher2 = []
                    for i in range(len(action_teacher)):
                        action_teacher2.append(action_teacher[i])
                    action_teacher2.pop()

                    #action_teacher = [action_teacher[0]]

                    difference = action_teacher2 - action
                    difference = np.array(difference)

                    randomNumber = random.random()
                    if t_total < 80500:

                        P_h = alpha * math.exp(-1 * tau * t_total)
                    else:
                        P_h = 0.0


                    if randomNumber < P_h:
                        h = [0] * dim_a
                        for i, name in enumerate(h):

                            if abs(difference[i]) > theta:
                                h[i] = np.sign(difference[i])




            #if metaworld_env:
            if info['success'] == 0:
                environment_done = False
                if t % 10 == 0:
                    print('fail!')



            else:
                success_counter += 1

                success_this_episode = 1
                environment_done = True
                if t % 10 == 0:
                    print('success!')


            # Compute done
            done = environment_done or repetition_is_over or t == max_time_steps_episode or doneButton #or feedback_joystick.ask_for_done()





            if np.any(h):
                h_counter += 1


            # Update weights
            if not evaluation and done == False:

                if agent.human_model_included:

                    agent.TRAIN_Human_Model_included(neural_network, action, t_total, done, i_episode, h, observation1, cummulative_feedback, t_total)
                else:
                    agent.TRAIN_Human_Model_NOT_included(neural_network, action, t_total, done, i_episode, h, observation1)



            t_total += 1



            # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
            r += reward




            # End of episode

            if done and (i_episode % 5 != 0):
                break

            if done and (i_episode % 5 == 0):

                print('HELLOOOOOOO')
                print('t total', t_total)

                success_this_episode = 0

                plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-v2-goal-observable"]
                env = plate_slide_goal_observable_cls()
                observation = env.reset()

                # Iterate over the episode
                for t_ev in range(0, max_time_steps_episode):



                    if t_ev % 10 == 0:
                        print('t_ev: ', t_ev)


                    #env.render(mode='human')

                    if metaworld_env:
                        observation_original = observation
                        # print("observation_original", observation_original)
                        # What is the useful part of the observation
                        if task == "drawer-open-v2-goal-observable":
                            observation = np.hstack((observation[:3], observation[4:7]))
                        elif task == "button-press-v2-goal-observable":
                            observation = np.hstack((observation[:3], observation[3], observation[4:7]))
                        elif task == "button-press-topdown-v2-goal-observable":
                            observation = np.hstack((observation[:3], observation[3], observation[4:7]))
                        elif task == "reach-v2-goal-observable":
                            observation = np.hstack((observation[:3], observation[3], observation[-3:]))
                        elif task == "plate-slide-v2-goal-observable":
                            observation = np.hstack(
                                (observation[:3], observation[3], observation[4:7], observation[-3:]))

                    # Prepare observation

                    observation = [observation]
                    observation1 = tf.convert_to_tensor(observation, dtype=tf.float32)

                    if kuka_env:
                        observation1 = tf.reshape(observation1, [1, agent.dim_o])
                    else:
                        observation1 = observation1

                        # Get action from the agent
                    action = agent.action(observation1)

                    # For the pressing button topdown task:
                    action_append_gripper = np.append(action, [1])

                    # Act
                    observation, reward, environment_done, info = env.step(action_append_gripper)

                    # if metaworld_env:
                    if info['success'] == 0:
                        environment_done = False


                    else:
                        success_counter += 1

                        success_this_episode = 1
                        environment_done = True


                    # Compute done
                    done_evaluation = environment_done or repetition_is_over or t_ev == 498 or doneButton  # or feedback_joystick.ask_for_done()

                    if done_evaluation:




                        print("\n")
                        print('%%% END OF EPISODE %%%')


                        total_r += r
                        cummulative_feedback = cummulative_feedback + h_counter





                        print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                        print('Successful episodes/total episodes: ', success_counter / episode_counter*100, '%')
                        print('Timesteps of this episode: ', t)
                        print('Timesteps of this repetition: ', t_total)

                        total_reward.append(r)
                        total_feedback.append(h_counter/(t + 1e-6))
                        total_success.append(success_counter)
                        total_success_div_episode.append(success_counter / episode_counter)
                        total_episodes.append(episode_counter)
                        total_time_steps.append(t_total)
                        total_t.append(t)
                        total_secs = time.time() - init_time
                        total_time_seconds.append(total_secs)
                        total_time_minutes.append(total_secs / 60)
                        total_cummulative_feedback.append(cummulative_feedback)
                        show_e.append(e)
                        show_buffer_size.append(buffer_size_max)
                        show_human_model.append(agent.human_model_included)
                        show_tau.append(tau)
                        total_task.append(task_short)
                        total_success_per_episode.append(success_this_episode)







                        if (save_results):


                            # Export data for plot
                            numpy_data = np.array([total_episodes, total_time_steps, total_reward, total_feedback, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_success_per_episode, total_t])
                            df = pd.DataFrame(data=numpy_data, index=["Episode", "Accumulated time steps", "Episode reward", "Episode feedback", "total seconds", "total minutes", "cummulative feedback", "e", "buffer size", "human model", "tau", "total_success", "total_success_div_episode", "success_this_episode", "timesteps_this_episode"])




                            path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                               '_e-' + str(e) + \
                                               '_B-' + str(buffer_size_max) + \
                                               '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short  +'_rep-rand-init-long-' + str(results_counter).zfill(2) + \
                                               '.csv'


                            if overwriteFiles == False:
                                while os.path.isfile(path_results):
                                    results_counter += 1
                                    path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                               '_e-' + str(e) + \
                                               '_B-' + str(buffer_size_max) + \
                                               '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short +'_rep-rand-init-long-' + str(results_counter).zfill(2) + \
                                               '.csv'


                            df.to_csv('./results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                               '_e-' + str(e) + \
                                               '_B-' + str(buffer_size_max) + \
                                               '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short + '_rep-rand-init-long-' + str(results_counter).zfill(2) + \
                                               '.csv', index=True)


                        env.close()
                        print('BREAAAAAAAAAAAAAAAAAAAAAAAK IIIIIIIIII')
                        break
                env.close()
                print('BREAAAAAAAAAAAAAAAAAAAAAAAK OOOOOOOOOOOOOO')
                break

