
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from collections import Counter
import glob

def postProcess(test_name):
    path = './results/'
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
        #print('timesteps_list0', timesteps_list)
        return_list.append(frame_to_list[14])
        counter = Counter(frame_to_list[14])
        print('counter: ', counter)

        feedback_list.append(frame_to_list[2])
        time_min_list.append(frame_to_list[4])
        # NEW ####
        policy_loss_agent_list.append(frame_to_list[12])
        policy_loss_hm_list.append(frame_to_list[13])
        ##########







    timesteps_list_crop_t = []
    return_list_crop_t = []
    feedback_list_crop_t = []
    time_min_list_crop_t = []

    # NEW ####
    policy_loss_agent_list_crop_t = []
    policy_loss_hm_list_crop_t = []
    ##########

    for t, r, f, t_min, pl_ag, pl_hm in zip(timesteps_list, return_list, feedback_list, time_min_list, policy_loss_agent_list, policy_loss_hm_list):
    #for t, r, f, t_min in zip(timesteps_list, return_list, feedback_list, time_min_list):

        fun_r = interp1d(t, r, fill_value="extrapolate")
        fun_f = interp1d(t, f, fill_value="extrapolate")
        # NEW ####
        fun_pl_ag = interp1d(t, pl_ag, fill_value="extrapolate")
        fun_pl_hm = interp1d(t, pl_hm, fill_value="extrapolate")
        ##########
        fun_t_min = interp1d(t, t_min, fill_value="extrapolate")
        timesteps_last_episode = t[-1]
        timesteps_last_episode = np.int64(timesteps_last_episode)

        xnew = np.linspace(0, timesteps_last_episode, num=timesteps_last_episode, endpoint=False)
        xnew = xnew.astype(int)

        #print('xnew: ', xnew)
        timesteps_list_crop_t.append(xnew)

        return_list_crop_t.append(fun_r(xnew))
        #print("Success list: ", return_list_crop_t)
        # NEW ####
        policy_loss_agent_list_crop_t.append(fun_pl_ag(xnew))
        policy_loss_hm_list_crop_t.append(fun_pl_hm(xnew))
        ##########
        feedback_list_crop_t.append(fun_f(xnew))
        time_min_list_crop_t.append(fun_t_min(xnew))
        #print('fun_t_min(xnew): ', fun_t_min(xnew))

    #print('timesteps_list_crop_t',timesteps_list_crop_t)
    max_timestep = []
    for item in timesteps_list_crop_t:

        max_timestep.append(max(item))


    min_timestep = min(item for item in max_timestep)


    min_timestep = min_timestep.item()

    mean_list = []
    feedback_list = []
    time_min_list = []

    # NEW ####
    pl_ag_mean_list = []
    pl_hm_mean_list = []
    ##########


    for t_list, r_list, f_list, t_min_list, pl_ag_list, pl_hm_list in zip(timesteps_list_crop_t, return_list_crop_t, feedback_list_crop_t, time_min_list_crop_t, policy_loss_agent_list_crop_t, policy_loss_hm_list_crop_t):
    #for t_list, r_list, f_list, t_min_list in zip(timesteps_list_crop_t, return_list_crop_t,feedback_list_crop_t,time_min_list_crop_t):

        # print(item2)

        t_list_list = t_list.tolist()
        r_list_list = r_list.tolist()
        f_list_list = f_list.tolist()


        # NEW ####
        pl_ag_list_list = pl_ag_list.tolist()
        pl_hm_list_list = pl_hm_list.tolist()
        ##########

        time_min_list_list = t_min_list.tolist()

        min_index_location = t_list_list.index(min_timestep)

        t_list_ok = t_list_list[:len(t_list_list) - (len(t_list_list) - min_index_location)]
        r_list_ok = r_list_list[:len(r_list_list) - (len(r_list_list) - min_index_location)]
        f_list_ok = f_list_list[:len(f_list_list) - (len(f_list_list) - min_index_location)]
        # NEW ####
        pl_ag_list_ok = pl_ag_list_list[:len(pl_ag_list_list) - (len(pl_ag_list_list) - min_index_location)]
        pl_hm_list_ok = pl_hm_list_list[:len(pl_hm_list_list) - (len(pl_hm_list_list) - min_index_location)]
        ##########
        t_min_list_ok = time_min_list_list[:len(time_min_list_list) - (len(time_min_list_list) - min_index_location)]


        mean_list.append(r_list_ok)
        feedback_list.append(f_list_ok)
        time_min_list.append(t_min_list_ok)

        mean_list.append(r_list_ok)
        # NEW ####
        pl_ag_mean_list.append(pl_ag_list_ok)
        pl_hm_mean_list.append(pl_hm_list_ok)
        ##########





    #print("mean",mean_list)



    #return t_list_ok, mean_list, feedback_list, time_min_list, min_index_location, e, buffer_size, human_model, tau
    return t_list_ok, mean_list, feedback_list, time_min_list, min_index_location, e, buffer_size, human_model, tau,  pl_ag_mean_list,  pl_hm_mean_list