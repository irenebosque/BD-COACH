
success_this_episode = 0


plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["plate-slide-v2-goal-observable"]
env = plate_slide_goal_observable_cls()
observation = env.reset()


# Iterate over the episode
for t_ev in range(1, max_time_steps_episode + 1):


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
            observation = np.hstack((observation[:3], observation[3], observation[4:7], observation[-3:]))

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
        print("Success!!!")
        success_this_episode = 1
        environment_done = True
        print('DONE')

    # Compute done
    done_evaluation = environment_done or repetition_is_over or t_ev == max_time_steps_episode or doneButton  # or feedback_joystick.ask_for_done()
