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



time.sleep(2)


repetition_is_over = False

if evaluation:
    print('***Evaluation TRUE: Load weights***')

    # test00 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-00.npy',
    #     allow_pickle=True)
    # test01 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-01.npy',
    #     allow_pickle=True)
    # test02 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-02.npy',
    #     allow_pickle=True)
    #
    # test03 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-03.npy',
    #     allow_pickle=True)
    # test04 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-04.npy',
    #     allow_pickle=True)
    # test05 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-05.npy',
    #     allow_pickle=True)
    #
    # test06 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-06.npy',
    #     allow_pickle=True)
    # test07 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-07.npy',
    #     allow_pickle=True)
    # test08 = np.load(
    #     './weights/weights-DCOACH_HM-False_e-0.1_B-10004_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-08.npy',
    #     allow_pickle=True)

    # testkuka00 = np.load(
    #     './weights/weights-DCOACH_HM-True_e-0.3_B-10000_tau-0.0001_lr-0.003_HMlr-0.0003_task-kuka-park-cardboard_rep-05.npy',
    #     allow_pickle=True)
    testmeta00 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.01_B-15000_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-02.npy',
        allow_pickle=True)
    testmeta01 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10009_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-01.npy',
        allow_pickle=True)
    testmeta02 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10009_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-02.npy',
        allow_pickle=True)
    testmeta03 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10009_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-03.npy',
        allow_pickle=True)
    testmeta04 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10009_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-04.npy',
        allow_pickle=True)
    testmeta05 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10009_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-05.npy',
        allow_pickle=True)
    testmeta06 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-06.npy',
        allow_pickle=True)
    testmeta07 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-07.npy',
        allow_pickle=True)
    testmeta08 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-08.npy',
        allow_pickle=True)
    testmeta09 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-09.npy',
        allow_pickle=True)
    testmeta10 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-10.npy',
        allow_pickle=True)
    testmeta11 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-11.npy',
        allow_pickle=True)
    testmeta12 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-12.npy',
        allow_pickle=True)
    testmeta13 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-13.npy',
        allow_pickle=True)
    testmeta14 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-14.npy',
        allow_pickle=True)
    testmeta15 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-15.npy',
        allow_pickle=True)
    testmeta16 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-16.npy',
        allow_pickle=True)
    testmeta17 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-17.npy',
        allow_pickle=True)
    testmeta18 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-18.npy',
        allow_pickle=True)
    testmeta19 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-19.npy',
        allow_pickle=True)
    testmeta20 = np.load(
        './weights/weights-DCOACH_HM-True_e-0.1_B-10007_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-20.npy',
        allow_pickle=True)



    #tests = [test00, test01, test02]
    #tests = [test03, test04, test05]
    #tests = [test06, test07, test08]


    #tests = [testmeta00, testmeta20]

    path = './weights/'
    generic_name = 'weights-DCOACH_HM-False_e-0.01_B-15000_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-'
    #tests_numbers = ["15", "16", "17", "18", "19", "20", "21", "22"]
    tests_numbers = ["15", "16", "17", "18", "19"]
    tests = loadWeights(path, generic_name, tests_numbers)

if evaluation:
    number_of_repetitions = len(tests)
else:
    number_of_repetitions = number_of_repetitions



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

    if evaluation:
        print('***Evaluation TRUE: weights of this repetition***')
        weigths_this_repetition = tests[i_repetition]

    episode_counter_5 = 0
    # Start training loop
    for i_episode in range(0, 399):
        print('range(max_num_of_episodes): ', range(max_num_of_episodes))
        success_this_episode = 0
        episode_counter += 1

        if kuka_env:
            if rospy.is_shutdown():
                print('shutdown')
                break

        if evaluation:
            print('\nHUman model included: ', str(agent.human_model_included) , ' buffer size: ', str(buffer_size_max) , ' e: ',  str(e), ' Repetition number:', i_repetition + 1, ' of ', len(tests), ' and episode ', i_episode, ' of ', len(weigths_this_repetition))
            print('weigths_this_repetition len: ', len(weigths_this_repetition))
            weigths_this_episode = weigths_this_repetition[i_episode]
            #weigths_this_episode = weigths_this_repetition[-1]
            #episode_counter_5
            #weigths_this_episode = weigths_this_repetition[episode_counter_5]
            #print("episode_counter_5: ", episode_counter_5)
            #episode_counter_5 += 5

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

        doneButton = False

        observation = env.reset()




        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(1, max_time_steps_episode+1):


            # if t_total == 1:
            #     print("Load initial weights!")
            #     agent.policy_model.set_weights(testkuka00[-1])

            init_time_timestep = time.time()
            if kuka_env:
                if rospy.is_shutdown():
                    print('shutdown')
                    break

            # if kuka_env == False:
            #     if render:
            #         #env.render()  # Make the environment visible
            #
            #         # obs_array = env.render(mode='rgb_array')
            #         # print("obs: ", obs_array.shape)
            #         # plt.imshow(obs_array)
            #         # #plt.figimage(image_to_show)
            #         # plt.draw()
            #         # plt.pause(0.00001)
            #
            #         env.render(mode='human')
            #         time.sleep(render_delay)  # Add delay to rendering if necessary


            h = None

            # Time the training takes
            secs = time.time() - init_time
            time_plot = str(datetime.timedelta(seconds=secs))

            # Print info
            print("\n")
            print('Rep: ', i_repetition, 'Episode: ', i_episode, "t this ep: ", t, ' total timesteps: ', t_total, 'time: ', time_plot)

            # Finish repetition if the maximum number of steps per repetition is reached
            if t_total == (max_time_steps_per_repetition):
                print('repetition is over!')
                repetition_is_over = True
                break

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
            print("action: ", action)

            # For the pressing button topdown task:
            action_append_gripper = np.append(action, [1])






            # Act
            observation, reward, environment_done, info = env.step(action_append_gripper)

            if evaluation:

                if kuka_env:
                    _, doneButton = env.get_h()
                    print("doneButton", doneButton)

            if not evaluation:

                # Get heedback h from a real human or from an oracle
                if human_teacher:


                    if kuka_env:
                        h, doneButton = env.get_h()
                        print("doneButton", doneButton)


                    elif cartpole_env or pendulum_env or mountaincar_env:

                        h = feedback_joystick_ROS.get_h()


                    for i, name in enumerate(h):
                        if abs(h[i]) < h_threshold:
                            h[i] = 0
                        else:
                            h[i] = h[i]


                    # if abs(h[0]) < 0.1:
                    #     h[0] = None
                    # elif abs(h[0]) > 0.1:
                    #     h[0] = h[0]


                    print('h: ', h)

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
                    if t_total < 25000:

                        P_h = alpha * math.exp(-1 * tau * t_total)
                    else:
                        P_h = 0.0


                    if randomNumber < P_h:
                        h = [0] * dim_a
                        for i, name in enumerate(h):

                            if abs(difference[i]) > theta:
                                h[i] = np.sign(difference[i])



                    # h = [0] * dim_a
                    # for i, name in enumerate(h):
                    #     if abs(difference[i]) > theta:
                    #         h[i] = np.sign(difference[i])


            #if metaworld_env:
            if info['success'] == 0:
                environment_done = False


            else:
                success_counter += 1
                print("Success!!!")
                success_this_episode = 1
                environment_done = True
                print('DONE')

            # h = env.get_h()
            # if h == True:
            #     done = True

            # Compute done
            done = environment_done or t == max_time_steps_episode or doneButton #or feedback_joystick.ask_for_done()
            # force it to do the max steps of a episode
            done =  t == max_time_steps_episode or doneButton  # or feedback_joystick.ask_for_done()
            # During evaluation of real kuka you can terminate an episode:
            print('doneeeeeeeeeeeeeeeeeeeeeeee: ', done)




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


            if kuka_env:
                rate = rospy.Rate(10)
                rate.sleep()
                print("sleep rate: ", rate)
            timee = time.time() - init_time_timestep
            print('timestepp: ', timee)

            # End of episode
            if done:
                #plt.show()
                print("\n")
                print('%%% END OF EPISODE %%%')
                #if evaluation:

                total_r += r
                cummulative_feedback = cummulative_feedback + h_counter
                if not evaluation:
                    print("cummulative fedback:", cummulative_feedback)
                    policy_loss_hm = 0
                    if cummulative_feedback > 20:

                        policy_loss_hm = agent.policy_loss_hm_batch
                        policy_loss_agent = agent.policy_loss_agent_batch
                    else:
                        #policy_loss_hm = 0
                        policy_loss_agent = 0



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
                if not evaluation:
                    total_policy_loss_agent.append(policy_loss_agent)
                    total_policy_loss_hm.append(policy_loss_hm)
                else:
                    total_policy_loss_agent.append(0)
                    total_policy_loss_hm.append(0)






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
                    numpy_data = np.array([total_time_steps, total_reward, total_feedback, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau, total_success, total_success_div_episode, total_policy_loss_agent, total_policy_loss_hm, total_success_per_episode, total_t])
                    df = pd.DataFrame(data=numpy_data, index=["Accumulated time steps", "Episode reward", "Episode feedback", "total seconds", "total minutes", "cummulative feedback", "e", "buffer size", "human model", "tau", "total_success", "total_success_div_episode", "total_policy_loss_agent", "total_policy_loss_hm", "success_this_episode", "timesteps_this_episode"])

                    # numpy_data = np.array(
                    #     [total_time_steps, total_reward, total_feedback, total_time_seconds, total_time_minutes,
                    #      total_cummulative_feedback, show_e, show_buffer_size, show_human_model, show_tau,
                    #      total_success, total_success_div_episode])
                    # df = pd.DataFrame(data=numpy_data,
                    #                   index=["Accumulated time steps", "Episode reward", "Episode feedback",
                    #                          "total seconds", "total minutes", "cummulative feedback", "e",
                    #                          "buffer size", "human model", "tau", "total_success",
                    #                          "total_success_div_episode"])


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

