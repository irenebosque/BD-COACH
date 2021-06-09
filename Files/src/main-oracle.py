import numpy as np
import pandas as pd
import time
import tensorflow as tf
import os
import random
import math
from main_init import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, tau, alpha, theta
save_results = True

"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""



# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)


# Instantiate model for the policy teacher
policy_teacher = neural_network.policy_model()
# Load the weights
policy_teacher.load_weights('./weights/TEACHER_OK')

e = 1
buffer_size_max = 1000
max_time_steps_per_repetition = 20000
number_of_repetitions = 20
repetition_is_over = False
evaluation = False
tau1 = 0.0007

results_counter = 0
weigths_counter = 0


agent.conditionTest(e, buffer_size_max)



time.sleep(2)


if evaluation:

    rep00 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-00.npy', allow_pickle=True)
    rep01 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-01.npy', allow_pickle=True)
    rep02 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-02.npy', allow_pickle=True)
    rep03 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-03.npy', allow_pickle=True)
    rep04 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-04.npy', allow_pickle=True)
    rep05 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-05.npy', allow_pickle=True)
    rep06 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-06.npy', allow_pickle=True)
    rep07 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-07.npy', allow_pickle=True)
    rep08 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-08.npy', allow_pickle=True)
    rep09 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-09.npy', allow_pickle=True)
    rep10 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-10.npy', allow_pickle=True)
    rep11 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-11.npy', allow_pickle=True)
    rep12 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-12.npy', allow_pickle=True)
    rep13 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-13.npy', allow_pickle=True)
    rep14 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-14.npy', allow_pickle=True)
    rep15 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-15.npy', allow_pickle=True)
    rep16 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-16.npy', allow_pickle=True)
    rep17 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-17.npy', allow_pickle=True)
    rep18 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-18.npy', allow_pickle=True)
    rep19 = np.load('./weights/weights-learning-policy-Human_model_included-True-e-0.001-buffer-size-1000_repetition-19.npy', allow_pickle=True)

    tests = [rep00,
             rep01,
             rep02,
             rep03,
             rep04,
             rep05,
             rep06,
             rep07,
             rep08,
             rep09,
             rep10,
             rep11,
             rep12,
             rep13,
             rep14,
             rep15,
             rep16,
             rep17,
             rep18,
             rep19]

    tests = [rep19]

for i_repetition in range(number_of_repetitions):



    # Initialize variables
    total_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model = [0.8], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included]
    t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback = 1, 0, 0, 0, 0, 0, 0
    human_done, random_agent, evaluation_started = False, False, False
    repetition_list = []


    init_time = time.time()

    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%  GENERALPARAMTERS OF THE TEST %%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('\nRepetition number:', i_repetition )
    print('Environment:', env)
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

        print('Starting episode number: ', i_episode)


        observation = env.reset()  # reset environment at the beginning of the episode


        past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


        # Iterate over the episode
        for t in range(int(max_time_steps_episode)):

            # Finish repetition if the maximum number of steps per repetition is reached

            if t_total == max_time_steps_per_repetition:
                repetition_is_over = True
                break




            if (t % 10 == 0):
                print("t this episode: ", t, " and t this repetition: ", t_total)

            h = None


            if render:
                env.render()  # Make the environment visible
                time.sleep(render_delay)  # Add delay to rendering if necessary

            # Transform observation so you can pass them to the neural network model
            observation = np.reshape(observation, [1, 2])
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)

            # Get feedback signal from real Human
            #h = human_feedback.get_h()
            #evaluation = human_feedback.evaluation

            if evaluation == False:
                # Get feedback from Oracle
                action_teacher = policy_teacher([observation])
                action_agent = agent.policy_model([observation])

                difference = action_teacher - action_agent

                randomNumber = random.random()
                P_h = 0.8 * math.exp(-1*tau1 * t_total)
                if randomNumber < P_h:

                    if difference > 0.1 or difference < -0.1:
                        # print('Oracle, gimme feedback!')
                        h = np.sign(difference)







            # Feed h to agent

            agent.feed_h(h)





            # Map action from observation
            #state_representation = transition_model.get_state_representation(neural_network, observation,  i_episode, t)
            action = agent.action(neural_network, observation, i_episode, t)



            # Act
            observation, reward, environment_done, info = env.step(action)


            # Compute done
            done = human_feedback.ask_for_done() or environment_done

            # Compute new hidden state of LSTM
            #transition_model.compute_lstm_hidden_state(neural_network, action)

            # Append transition to database
            if not evaluation:
                if past_action is not None and past_observation is not None:
                    #episode_trajectory.append([past_observation, past_action, transition_model.processed_observation])  # append o, a, o' (not really necessary to store it like this)
                    episode_trajectory.append([past_observation, past_action])
                #past_observation, past_action = transition_model.processed_observation, action
                past_observation, past_action = observation, action

                if t % 100 == 0 or done:
                    trajectories_database.append(episode_trajectory)  # append episode trajectory to database
                    episode_trajectory = []

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
                print('%%% END OF EPISODE %%%')
                #if evaluation:

                total_r += r
                cummulative_feedback = cummulative_feedback + h_counter



                print('Episode Reward:', '%.3f' % r)
                print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                total_reward.append(r)
                total_feedback.append(h_counter/(t + 1e-6))
                total_time_steps.append(t_total)
                total_secs = time.time() - init_time
                total_time_seconds.append(total_secs)
                total_time_minutes.append(total_secs / 60)
                total_cummulative_feedback.append(cummulative_feedback)
                show_e.append(e)
                show_buffer_size.append(buffer_size_max)
                show_human_model.append(agent.human_model_included)



                if (save_results):
                    print('total_reward: ', total_reward)
                    print('total_feedback: ', total_feedback)
                    print('total_time_steps: ', total_time_steps)
                    print('total_time_seconds: ', total_time_seconds)
                    print('total_time_minutes: ', total_time_minutes)
                    print('cummulative_feedback: ', total_cummulative_feedback)




                    ############################################################

                    # Export data for plot
                    numpy_data = np.array([total_time_steps, total_reward, total_feedback, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model])
                    df = pd.DataFrame(data=numpy_data, index=["Accumulated time steps", "Episode reward", "Episode feedback", "total seconds", "total minutes", "cummulative feedback", "e", "buffer size", "human model"])

                    path_results = './results/DCOACH_' + 'Human_model_included-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_buffer-' + str(buffer_size_max) + \
                                   '_Evaluation-'+ str(evaluation) + '_repetition-' + str(results_counter).zfill(2) + \
                                       '.csv'

                    print('check this path: ', path_results)
                    print('results_counter: ', results_counter)
                    if overwriteFiles == False:
                        while os.path.isfile(path_results):
                            results_counter += 1
                            path_results = './results/DCOACH_' + 'Human_model_included-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_buffer-' + str(buffer_size_max) + \
                                       '_Evaluation-'+ str(evaluation) + '_repetition-' + str(results_counter).zfill(2) + \
                                       '.csv'
                            print('results counter: ', results_counter)

                    df.to_csv('./results/DCOACH_' + 'Human_model_included-' + str(agent.human_model_included) + \
                                       '_e-' + str(e) + \
                                       '_buffer-' + str(buffer_size_max) + \
                                       '_Evaluation-'+ str(evaluation) + '_repetition-' + str(results_counter).zfill(2) + \
                                       '.csv', index=True)

                    if not evaluation:
                        repetition_list.append(agent.policy_model.get_weights())
                        repetition_list_np_array = np.array(repetition_list)

                        path_weights = './weights/weights-learning-policy-' + 'Human_model_included-' + str(agent.human_model_included) +\
                            '-e-' + str(e) + \
                            '-buffer-size-' + str(buffer_size_max) + \
                            '_repetition-' + str(weigths_counter).zfill(2) + '.npy'

                        if overwriteFiles == False:
                            while os.path.isfile(path_weights):
                                weigths_counter += 1
                                path_weights = './weights/weights-learning-policy-' + 'Human_model_included-' + str(agent.human_model_included) +\
                            '-e-' + str(e) + \
                            '-buffer-size-' + str(buffer_size_max) + \
                            '_repetition-' + str(weigths_counter).zfill(2) + '.npy'

                        np.save(
                            './weights/weights-learning-policy-' + 'Human_model_included-' + str(agent.human_model_included) +\
                            '-e-' + str(e) + \
                            '-buffer-size-' + str(buffer_size_max) + \
                            '_repetition-' + str(weigths_counter).zfill(2) + '.npy', repetition_list_np_array)



                if render:
                    time.sleep(1)

                print('Total time (s):', '%.3f' % (time.time() - init_time))


                break