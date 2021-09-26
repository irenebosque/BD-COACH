if done and (i_episode % 5 == 0):
    for i_ev in range(0, evaluations_per_training):

        # Iterate over the episode
        for t_ev in range(0, max_time_steps_episode):

            # Act
            observation, reward, environment_done, info = env.step(action_append_gripper)

            if done_evaluation:

                if (save_results):


                    if overwriteFiles == False:
                        while os.path.isfile(path_results):
                            results_counter += 1
                            path_results = './results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                                           '_e-' + str(e) + \
                                           '_B-' + str(buffer_size_max) + \
                                           '_tau-' + str(tau) + '_lr-' + str(lr) + '_HMlr-' + str(
                                HM_lr) + '_agent_batch_lr-' + str(
                                agent_with_hm_learning_rate) + '_task-' + task_short + '_rep-randm-' + str(
                                results_counter).zfill(2) + \
                                           '.csv'

                    df.to_csv('./results/DCOACH_' + 'HM-' + str(agent.human_model_included) + \
                              '_e-' + str(e) + \
                              '_B-' + str(buffer_size_max) + \
                              '_tau-' + str(tau) + '_lr-' + str(lr) + '_HMlr-' + str(HM_lr) + '_agent_batch_lr-' + str(
                        agent_with_hm_learning_rate) + '_task-' + task_short + '_rep-randm-' + str(results_counter).zfill(
                        2) + \
                              '.csv', index=False)

                break

    break



