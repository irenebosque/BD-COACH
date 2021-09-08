
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
    policy_loss_agent_list = []
    policy_loss_hm_list = []
    feedback_list = []
    time_min_list = []

    for i in range(0, repetition_counter):

        d_frame = pd.read_csv(path + test_name.format(str(
                                i).zfill(2)))

        frame_to_list = d_frame.values.tolist()

        del frame_to_list[0][0]
        del frame_to_list[14][0]
        del frame_to_list[2][0]
        del frame_to_list[4][0]
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
        #task = frame_to_list[12][1]
        #print('task: ', task)





        timesteps_list.append(frame_to_list[0])
        return_list.append(frame_to_list[14])
        feedback_list.append(frame_to_list[2])

    print("return_list: ", return_list)
    a = np.array(return_list)
    return_list = np.mean(a, axis =0)
    a = np.array(timesteps_list)
    timesteps_list = np.mean(a, axis=0)
    print("return_list: ", len(return_list))
    print("timesteps_list: ", len(timesteps_list))

    z = np.polyfit(timesteps_list, return_list, 6)

    p = np.poly1d(z)
    print("---------")
    print("p: ", z)
    # ,

    #return t_list_ok, mean_list, feedback_list, time_min_list, min_index_location, e, buffer_size, human_model, tau
    #return timesteps_list, p(timesteps_list), feedback_list[0], tau, e, human_model
    return timesteps_list, return_list, feedback_list[0], tau, e, human_model

