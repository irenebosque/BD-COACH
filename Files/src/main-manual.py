import numpy as np
import pandas as pd
import time
import tensorflow as tf
import os
import random
import math


from main_init import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num,count_down, \
                        max_num_of_episodes, env, render, max_time_steps_episode, human_feedback_keyboard, save_results, eval_save_path, \
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




e = 1
buffer_size_max = 1000
evaluation = False
tau1 = 0.0007




agent.conditionTest(e, buffer_size_max)



time.sleep(2)




# Initialize variables
total_feedback, total_time_steps, trajectories_database, total_reward, total_time_seconds, total_time_minutes, total_cummulative_feedback, show_e, show_buffer_size, show_human_model = [0.8], [0], [], [0], [0], [0], [0], [e], [buffer_size_max], [agent.human_model_included]
t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r, cummulative_feedback = 1, 0, 0, 0, 0, 0, 0
human_done, random_agent, evaluation_started = False, False, False



init_time = time.time()

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%  GENERALPARAMTERS OF THE TEST %%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

print('\nEnvironment:', env)
print('Learning algorithm: ', agent_type)
print('Evaluation?: ', evaluation)
print('e: ' + str(e) + ' and ' + 'buffer size: ' + str(buffer_size_max))
print('Human model included?: ' + str(agent.human_model_included) + '\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')


agent.createModels(neural_network)




# Start training loop
for i_episode in range(max_num_of_episodes):





    if i_episode == 0:
        overwriteFiles = False
    if i_episode != 0:
        overwriteFiles = True

    print('Starting episode number: ', i_episode)


    observation = env.reset()  # reset environment at the beginning of the episode


    past_action, past_observation, episode_trajectory, r, h_counter= None, None, [], 0, 0 # reset variables for new episode


    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):




        if (t % 10 == 0):
            print("t this episode: ", t, " and t this repetition: ", t_total)

        h = None


        if render:
            env.render()  # Make the environment visible
            time.sleep(render_delay)  # Add delay to rendering if necessary

        # Transform observation so you can pass them to the neural network model
        print('observation before: ', observation)
        observation = np.reshape(observation, [1, 2])
        print('observation after: ', observation)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # Get feedback signal from real Human
        h = human_feedback_keyboard.get_h()

        evaluation = human_feedback_keyboard.evaluation



        # Feed h to agent
        agent.feed_h(h)



        # Map action from observation
        action = agent.action(neural_network, observation, i_episode, t)



        # Act
        observation, reward, environment_done, info = env.step(action)


        # Compute done
        done = human_feedback_keyboard.ask_for_done() or environment_done



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





            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))


            break
