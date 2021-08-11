import numpy as np
import pandas as pd
import time
import datetime
import tensorflow as tf
import os
import random
from tabulate import tabulate
import rospy

import math
from main_init import neural_network, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, render, max_time_steps_episode, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta, task, max_time_steps_per_repetition, \
number_of_repetitions, evaluation, save_results, env, task_short, dim_a, mountaincar_env, metaworld_env, human_teacher, oracle_teacher, action_factor

e = agent.e
action_limit = agent.action_limit
buffer_size_max = agent.buffer_max_size
lr = agent.policy_model_learning_rate
HM_lr = agent.human_model_learning_rate


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



time.sleep(2)


repetition_is_over = False

if evaluation:
    print('***Evaluation TRUE: Load weights***')

    test00 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.000101_lr-0.005_HMlr-0.002_task-reach_rep-00.npy',
        allow_pickle=True)
    test01 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.000101_lr-0.005_HMlr-0.002_task-reach_rep-01.npy',
        allow_pickle=True)
    test02 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.000101_lr-0.005_HMlr-0.002_task-reach_rep-02.npy',
        allow_pickle=True)
    test03 = np.load(
        './weights/weights-DCOACH_HM-True_e-1.0_B-10000_tau-0.000101_lr-0.005_HMlr-0.002_task-reach_rep-03.npy',
        allow_pickle=True)




    tests = [test03]






# Initialize variables
total_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_episodes, total_task = [alpha], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included], [tau], [0], [0], [0], [task_short]
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
print('e: ' + str(e) + ' and ' + 'buffer size: ' + str(buffer_size_max))
print('Human model included?: ' + str(agent.human_model_included) + '\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')


agent.createModels(neural_network)

if evaluation:
    print('***Evaluation TRUE: weights of this repetition***')
    weigths_this_repetition = tests[i_repetition]


# Start training loop
for i_episode in range(max_num_of_episodes):
    episode_counter += 1
    if rospy.is_shutdown():
        print('shutdown')
        break

    if evaluation:
        print('\nHUman model included: ', str(agent.human_model_included) , ' buffer size: ', str(buffer_size_max) , ' e: ',  str(e), ' Repetition number:', i_repetition + 1, ' of ', len(tests), ' and episode ', i_episode + 1, ' of ', len(weigths_this_repetition))
        weigths_this_episode = weigths_this_repetition[i_episode]
        #weigths_this_episode = weigths_this_repetition[-1]

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
        if rospy.is_shutdown():
            print('shutdown')
            break


        h = None

        # Time the training takes
        secs = time.time() - init_time
        time_plot = str(datetime.timedelta(seconds=secs))

        # Print info
        print("\n")
        print('Episode: ', i_episode, "t this ep: ", t, ' total timesteps: ', t_total, 'time: ', time_plot)

        # Finish repetition if the maximum number of steps per repetition is reached
        if t_total == (max_time_steps_per_repetition):
            print('repetition is over!')
            repetition_is_over = True
            break

        # if render:
        #     env.render()  # Make the environment visible
        #     time.sleep(render_delay)  # Add delay to rendering if necessary



        # Prepare observation
        observation = [observation]
        observation1 = tf.convert_to_tensor(observation, dtype=tf.float32)
        observation1 = tf.reshape( observation1, [1, 6])

        # Get action from the agent
        action = agent.action(observation1)



        # Act
        observation, reward, environment_done = env.step(action)





        # Get heedback h from a real human or from an oracle
        if human_teacher:

            #h = feedback_joystick.get_h()
            h = env.get_h()


            if abs(h[0]) < 0.1:
                h[0] = None
            elif abs(h[0]) > 0.1:
                h[0] = h[0]


            print('h2: ', h)

        elif oracle_teacher:

            action_teacher = policy_oracle([observation1])
            action_teacher = np.clip(action_teacher, -1, 1)
            action_teacher = [action_teacher[0]]

            difference = action_teacher - action
            difference = np.array(difference)
            randomNumber = random.random()

            P_h = alpha * math.exp(-1 * tau * t_total)

            if randomNumber < P_h:
                if abs(difference) > theta:
                    h = np.sign(difference)
                else:
                    h = None


        # if metaworld_env:
        #     if info['success'] == 0:
        #         environment_done = False
        #
        #     else:
        #         success_counter += 1
        #         environment_done = True
        #         print('DONE')


        # Compute done
        done = environment_done or t == max_time_steps_episode #or feedback_joystick.ask_for_done()


        if np.any(h):
            h_counter += 1


        # Update weights
        if not evaluation:

            if agent.human_model_included:

                agent.TRAIN_Human_Model_included(neural_network, action, t_total, done, i_episode, h, observation1)
            else:
                agent.TRAIN_Human_Model_NOT_included2(neural_network, action, t_total, done, i_episode, h, observation1)



        t_total += 1



        # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
        r += reward

        rate = rospy.Rate(10)
        rate.sleep()

        # End of episode
        if done:
            print("\n")
            print('%%% END OF EPISODE %%%')
            #if evaluation:

            total_r += r
            cummulative_feedback = cummulative_feedback + h_counter



            #print('Episode Reward:', '%.3f' % r)
            #print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
            print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
            print('Success rate: ', success_counter / episode_counter)

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
            total_task.append(task_short)



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
                               '_Eval-'+ str(evaluation) +  '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short  +'_rep-' + str(results_counter).zfill(2) + \
                                   '.csv'


                if overwriteFiles == False:
                    while os.path.isfile(path_results):
                        results_counter += 1
                        path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                   '_e-' + str(e) + \
                                   '_B-' + str(buffer_size_max) + \
                                   '_Eval-'+ str(evaluation) +  '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short +'_rep-' + str(results_counter).zfill(2) + \
                                   '.csv'


                df.to_csv('./results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                   '_e-' + str(e) + \
                                   '_B-' + str(buffer_size_max) + \
                                   '_Eval-'+ str(evaluation) + '_tau-' + str(tau) +  '_lr-' + str(lr) +  '_HMlr-' + str(HM_lr)+ '_task-' + task_short + '_rep-' + str(results_counter).zfill(2) + \
                                   '.csv', index=True)


                if not evaluation:
                    repetition_list.append(agent.policy_model.get_weights())
                    repetition_list_np_array = np.array(repetition_list)



                    path_weights = './weights/weights-DCOACH_' + 'HM-' + str(agent.human_model_included) +\
                        '_e-' + str(e) + \
                        '_B-' + str(buffer_size_max) + \
                        '_tau-' + str(tau) + \
                        '_lr-' + str(lr) + \
                        '_HMlr-' + str(HM_lr) + \
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
                        '_HMlr-' + str(HM_lr) + \
                        '_task-' + task_short + \
                        '_rep-' + str(weigths_counter).zfill(2) + '.npy'



                    np.save(
                        './weights/weights-DCOACH_' + 'HM-' + str(agent.human_model_included) +\
                        '_e-' + str(e) + \
                        '_B-' + str(buffer_size_max) + \
                        '_tau-' + str(tau) + \
                        '_lr-' + str(lr) + \
                        '_HMlr-' + str(HM_lr) + \
                        '_task-' + task_short + \
                        '_rep-' + str(weigths_counter).zfill(2) + '.npy', repetition_list_np_array)







            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))

            #env.close()
            break

#rospy.spin()
