
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from collections import Counter
import glob

def postProcess(test_name, path):

    repetition_counter = 0

    test_name_rep_counter = test_name.replace("{}", "??")

    for name in glob.glob(path + test_name_rep_counter):
        repetition_counter += 1
        print('name: ', name)
    print('repetition_counter', repetition_counter)
    print('pruebita')
    timesteps_list = []
    return_list = []
    minutes_list = []
    policy_loss_agent_list = []
    policy_loss_hm_list = []
    feedback_list = []
    time_min_list = []

    for i in range(0, repetition_counter):

        d_frame = pd.read_csv(path + test_name.format(str(
                                i).zfill(2)))
        d_frame.iloc[1, -1] = 60500


        frame_to_list = d_frame.values.tolist()

        del frame_to_list[1][0] # timesteps
        del frame_to_list[13][0] # success / return
        del frame_to_list[3][0] # feedback
        del frame_to_list[5][0] # minutes
        # NEW ####

        ##########
        e = frame_to_list[7][-1]
        buffer_size = frame_to_list[8][-1]
        if frame_to_list[9][-1] == 1:
            human_model = 'yes'
        else:
            human_model = 'no'
        tau = frame_to_list[9][1]




        print('frame_to_list[0]', frame_to_list[0])

        timesteps_list.append(frame_to_list[1])

        return_list.append(frame_to_list[13])
        feedback_list.append(frame_to_list[3])
        minutes_list.append(frame_to_list[5])


    #print('timesteps_list.append(frame_to_list[0])', timesteps_list)

    timesteps_list_ok = []
    success_list_ok = []
    feedback_list_ok = []

    for t, s, f in zip(timesteps_list, return_list, feedback_list):

        fun_s = interp1d(t, s)
        fun_f = interp1d(t, f)

        timesteps_last_episode = t[-1]
        timesteps_last_episode = np.int64(timesteps_last_episode)

        xnew = np.linspace(0, timesteps_last_episode, num=timesteps_last_episode, endpoint=False)
        xnew = xnew.astype(int)

        success = fun_s(xnew)
        feedback = fun_f(xnew)

        # xnew = xnew[: len(xnew) - 1000]
        # success = success[: len(success) - 1000]
        # feedback  = feedback [: len(feedback ) - 1000]

        timesteps_list_ok.append(xnew)
        success_list_ok.append(success)
        feedback_list_ok.append(feedback )


    # timesteps_list_ok = timesteps_list_ok[0]
    # success_list_ok = success_list_ok[0]
    # feedback_list_ok = feedback_list_ok[0]

    # timesteps_list_ok = timesteps_list_ok.tolist()
    # success_list_ok = success_list_ok.tolist()
    print('timesteps: ', timesteps_list_ok[0])
    print('timesteps len : ', len(timesteps_list_ok[0]))





    return timesteps_list_ok, success_list_ok, feedback_list_ok, tau, e, human_model, policy_loss_agent_list, policy_loss_hm_list, buffer_size

