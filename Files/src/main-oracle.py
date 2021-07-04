import numpy as np
import pandas as pd
import time
import datetime
import tensorflow as tf
import os
import random
from tabulate import tabulate

import math
from main_init import neural_network, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, render, max_time_steps_episode, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta, task, max_time_steps_per_repetition, \
number_of_repetitions, evaluation, save_results, env, policy_oracle, task_short

e = agent.e
buffer_size_max = agent.buffer_max_size
lr = agent.policy_model_learning_rate




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



time.sleep(2)


repetition_is_over = False

if evaluation:

    repF00 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-00.npy',
        allow_pickle=True)
    repF01 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-01.npy',
        allow_pickle=True)
    repF02 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-02.npy',
        allow_pickle=True)
    repF03 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-03.npy',
        allow_pickle=True)
    repF04 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-04.npy',
        allow_pickle=True)
    repF05 = np.load(
        './weights/weights-DCOACH_HM-False_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-05.npy',
        allow_pickle=True)





    repT00 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-00.npy',
        allow_pickle=True)
    repT01 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-01.npy',
        allow_pickle=True)
    repT02 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-02.npy',
        allow_pickle=True)
    repT03 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-03.npy',
        allow_pickle=True)
    repT04 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-04.npy',
        allow_pickle=True)
    repT05 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.00016_lr-0.007_task-hockey_rep-05.npy',
        allow_pickle=True)

    #tests = [repF00, repF01, repF02]
    #tests = [repT00, repT01, repT02]
    tests = [repF02]


for i_repetition in range(number_of_repetitions):



    # Initialize variables
    total_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_episodes = [alpha], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included], [tau], [0], [0], [0]
    t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback, success_counter, episode_counter = 1, 0, 0, 0, 0, 0, 0, 0, 0
    human_done, random_agent, evaluation_started = False, False, False
    repetition_list = []


    init_time = time.time()

    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%  GENERALPARAMTERS OF THE TEST %%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('\nRepetition number:', i_repetition )
    print('Learning algorithm: ', agent_type)
    print('Evaluation?: ', evaluation)
    print('e: ' + str(e) + ' and ' + 'buffer size: ' + str(buffer_size_max))
    print('Human model included?: ' + str(agent.human_model_included) + '\n')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')


    agent.createModels(neural_network)

    if evaluation:
        weigths_this_repetition = tests[i_repetition]


    # Start training loop
    for i_episode in range(max_num_of_episodes):
        episode_counter += 1

        if evaluation:
            print('\nHUman model included: ', str(agent.human_model_included) , ' buffer size: ', str(buffer_size_max) , ' e: ',  str(e), ' Repetition number:', i_repetition + 1, ' of ', len(tests), ' and episode ', i_episode + 1, ' of ', len(weigths_this_repetition))
            weigths_this_episode = weigths_this_repetition[i_episode]
            agent.policy_model.set_weights(weigths_this_episode)


        if repetition_is_over == True:
            repetition_is_over = False
            break

        if i_episode == 0:
            overwriteFiles = False
        if i_episode != 0:
            overwriteFiles = True
        print("\n")
        print('Starting episode number: ', i_episode)


        observation = env.reset()



        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(int(max_time_steps_episode + 1)):

            # Finish repetition if the maximum number of steps per repetition is reached


            if t_total == (max_time_steps_per_repetition):
                print('repetition is over!')
                repetition_is_over = True
                break






            h = None


            if render:
                env.render()  # Make the environment visible
                time.sleep(render_delay)  # Add delay to rendering if necessary

            # to the oracle i pass the full observation:
            observation_original = observation

            # What is the useful part of the observation
            if task == "drawer-open-v2-goal-observable":
                observation = np.hstack((observation[:3], observation[4:7]))
            elif task == "button-press-v2-goal-observable":
                observation = np.hstack((observation[:3], observation[4:7]))
            elif task == "reach-v2-goal-observable":
                observation = np.hstack((observation[:3], observation[-3:]))
            elif task == "plate-slide-v2-goal-observable":
                observation = np.hstack((observation[:3], observation[4:7], observation[-3:]))

            print('observation', observation)

            if (t % 10 == 0 ):
                print("\n")
                data = [["obs_hand", observation[:3][0], observation[:3][1], observation[:3][2]],
                        ["obs_target", observation[-3:][0], observation[-3:][1], observation[-3:][2]]]

                print(tabulate(data, headers=["what", "dx", "dy", "dz"]))
                print("\n")









            observation = [observation]
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)



            # Get feedback signal from real Human
            #h = human_feedback.get_h()
            #evaluation = human_feedback.evaluation


            # Get action from the agent
            action = agent.action(observation)


            if evaluation == False:
                # Get action from Oracle
                action_teacher = policy_oracle.get_action(observation_original)
                action_teacher = np.clip(action_teacher, -1, 1)
                action_teacher = [action_teacher[0], action_teacher[1], action_teacher[2]]



                difference = action_teacher - action
                difference = np.array(difference)


                randomNumber = random.random()

                #P_h = alpha * math.exp(-1*tau * t_total)
                P_h = 0.6

                if randomNumber < P_h:

                    h = [0.0, 0.0, 0.0, 0.0]
                    if abs(difference[0]) > theta:
                        h[0] = np.sign(difference[0])

                    if abs(difference[1]) > theta:
                        h[1] = np.sign(difference[1])

                    if abs(difference[2]) > theta:
                        h[2] = np.sign(difference[2])


            '''
            if (t % 10 == 0 and np.any(h)):
                print("\n")
                print("t this episode: ", t, " and t this repetition: ", t_total)


                data = [["action_teacher", action_teacher[0], action_teacher[1], action_teacher[2]],
                        ["action_agent", action[0], action[1], action[2]],
                        ["difference", difference[0], difference[1], difference[2]],
                        ["h", h[0], h[1], h[2]]]

                print(tabulate(data, headers=["what", "dx", "dy", "dz", "gripper"]))
            '''


            if (t % 10 == 0 and i_episode>0):
                print("Task: ", task_short, ", Repetition: ", i_repetition, ", episode: ", i_episode, ", t this rep: ", t_total, ", success: ", (success_counter / episode_counter*100), "%", ", Total time: ", str(datetime.timedelta(seconds=total_secs)))


            # Feed h to agent
            agent.feed_h(h)





            # Map action from observation
            #state_representation = transition_model.get_state_representation(neural_network, observation,  i_episode, t)
            #action = agent.action(observation)




            # Act
            action_to_env = np.append(action, 0)
            observation, reward, environment_done, info = env.step(action_to_env)


            if info['success'] == 0:
                environment_done = False
            else:
                success_counter += 1
                environment_done = True


            # Compute done
            #done = human_feedback.ask_for_done() or environment_done
            done = environment_done or t == max_time_steps_episode


            if np.any(h):
                h_counter += 1


            # Update weights transition model/policy=
            if not evaluation:
                #if done:
                    #t_total = done  # tell the agents that the episode finished


                if agent.human_model_included:
                    agent.TRAIN_Human_Model_included(neural_network, action, t_total, done, i_episode)
                else:
                    agent.TRAIN_Human_Model_NOT_included(neural_network, action, t_total, done, i_episode)



            t_total += 1



            # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
            r += reward

            # End of episode
            if done:
                print("\n")
                print('%%% END OF EPISODE %%%')
                #if evaluation:

                total_r += r
                cummulative_feedback = cummulative_feedback + h_counter



                print('Episode Reward:', '%.3f' % r)
                print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))

                total_reward.append(r)
                total_feedback.append(h_counter/(t + 1e-6))
                total_success.append(success_counter)
                total_success_div_episode.append(success_counter / episode_counter)
                total_episodes.append(episode_counter)
                total_time_steps.append(t_total)
                total_secs = time.time() - init_time
                total_time_seconds.append(total_secs)
                total_time_minutes.append(total_secs / 60)
                total_cummulative_feedback.append(cummulative_feedback)
                show_e.append(e)
                show_buffer_size.append(buffer_size_max)
                show_human_model.append(agent.human_model_included)
                show_tau.append(tau)



                if (save_results):
                    '''
                    print('total_reward: ', total_reward)
                    print('total_feedback: ', total_feedback)
                    print('total_time_steps: ', total_time_steps)
                    print('total_time_seconds: ', total_time_seconds)
                    print('total_time_minutes: ', total_time_minutes)
                    print('cummulative_feedback: ', total_cummulative_feedback)                    
                    '''





                    ############################################################

                    # Export data for plot
                    numpy_data = np.array([total_time_steps, total_reward, total_feedback, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode])
                    df = pd.DataFrame(data=numpy_data, index=["Accumulated time steps", "Episode reward", "Episode feedback", "total seconds", "total minutes", "cummulative feedback", "e", "buffer size", "human model", "tau", "total_success", "total_success_div_episode"])

                    path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_B-' + str(buffer_size_max) + \
                                   '_Eval-'+ str(evaluation) +  '_tau-' + str(tau) +  '_lr-' + str(lr) + '_task-' + task_short +'_rep-' + str(results_counter).zfill(2) + \
                                       '.csv'



                    if overwriteFiles == False:
                        while os.path.isfile(path_results):
                            results_counter += 1
                            path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_B-' + str(buffer_size_max) + \
                                       '_Eval-'+ str(evaluation) +  '_tau-' + str(tau) +  '_lr-' + str(lr) + '_task-' + task_short +'_rep-' + str(results_counter).zfill(2) + \
                                       '.csv'


                    df.to_csv('./results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_B-' + str(buffer_size_max) + \
                                       '_Eval-'+ str(evaluation) + '_tau-' + str(tau) +  '_lr-' + str(lr) + '_task-' + task_short + '_rep-' + str(results_counter).zfill(2) + \
                                       '.csv', index=True)

                    
                    if not evaluation:
                        repetition_list.append(agent.policy_model.get_weights())
                        repetition_list_np_array = np.array(repetition_list)



                        path_weights = './weights/weights-DCOACH_' + 'HM-' + str(agent.human_model_included) +\
                            '_e-' + str(e) + \
                            '_B-' + str(buffer_size_max) + \
                            '_tau-' + str(tau) + \
                            '_lr-' + str(lr) + \
                            '_task-' + task_short + \
                            '_rep-' + str(weigths_counter).zfill(2) + '.npy'



                        if overwriteFiles == False:

                            while os.path.isfile(path_weights):

                                weigths_counter += 1
                                path_weights = './weights/weights-DCOACH_' + 'HM-' + str(agent.human_model_included) +\
                            '_e-' + str(e) + \
                            '_B-' + str(buffer_size_max) + \
                            '_tau-' + str(tau) + \
                            '_lr-' + str(lr) + \
                            '_task-' + task_short + \
                            '_rep-' + str(weigths_counter).zfill(2) + '.npy'



                        np.save(
                            './weights/weights-DCOACH_' + 'HM-' + str(agent.human_model_included) +\
                            '_e-' + str(e) + \
                            '_B-' + str(buffer_size_max) + \
                            '_tau-' + str(tau) + \
                            '_lr-' + str(lr) + \
                            '_task-' + task_short + \
                            '_rep-' + str(weigths_counter).zfill(2) + '.npy', repetition_list_np_array)
                    
                    





                if render:
                    time.sleep(1)

                print('Total time (s):', '%.3f' % (time.time() - init_time))

                #env.close()
                break