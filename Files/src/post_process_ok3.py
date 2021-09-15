
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


        frame_to_list = d_frame.values.tolist()

        del frame_to_list[0][0] # timesteps
        del frame_to_list[14][0] # success / return
        del frame_to_list[5][0] # feedback
        del frame_to_list[4][0] # minutes
        # NEW ####
        del frame_to_list[12][0]
        del frame_to_list[13][0]
        ##########
        e = frame_to_list[6][-1]
        buffer_size = frame_to_list[7][-1]
        if frame_to_list[8][-1] == 1:
            human_model = 'yes'
        else:
            human_model = 'no'
        tau = frame_to_list[9][1]






        timesteps_list.append(frame_to_list[0])

        return_list.append(frame_to_list[14])
        feedback_list.append(frame_to_list[5])
        minutes_list.append(frame_to_list[4])
        policy_loss_agent_list.append(frame_to_list[12])
        policy_loss_hm_list.append(frame_to_list[13])

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

        timesteps_list_ok.append(xnew)
        success_list_ok.append(fun_s(xnew))
        feedback_list_ok.append(fun_f(xnew))


    # timesteps_list_ok = timesteps_list_ok[0]
    # success_list_ok = success_list_ok[0]
    # feedback_list_ok = feedback_list_ok[0]

    # timesteps_list_ok = timesteps_list_ok.tolist()
    # success_list_ok = success_list_ok.tolist()




    return timesteps_list_ok, success_list_ok, feedback_list_ok, tau, e, human_model, policy_loss_agent_list, policy_loss_hm_list, buffer_size

