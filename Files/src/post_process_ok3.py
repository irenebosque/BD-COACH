
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from collections import Counter
import glob
from main_init import max_time_steps_per_repetition

def postProcess(test_name, path):

    repetition_counter = 0

    test_name_rep_counter = test_name.replace("{}", "??")

    for name in glob.glob(path + test_name_rep_counter):
        repetition_counter += 1
        print('name: ', name)
    print('repetition_counter', repetition_counter)



    timesteps_list    = []
    success_list      = []
    feedback_list     = []
    pct_feedback_list = []


    list_max_timestep = []
    for i in range(0, repetition_counter):

        df = pd.read_csv(path + test_name.format(str(i).zfill(2)))
        last_timestep = df.iloc[-1]['Timesteps']
        # print("len(list_max_timestep): ", len(list_max_timestep))
        # print("sum(list_max_timestep): ", sum(list_max_timestep))
        list_max_timestep.append(last_timestep)
    average_max_timestep = sum(list_max_timestep) / len(list_max_timestep)
    average_max_timestep = int(average_max_timestep)
    print(average_max_timestep)



    for i in range(0, repetition_counter):

        df = pd.read_csv(path + test_name.format(str(i).zfill(2)))


        df.at[df.tail(1).index.item(), 'Timesteps'] = average_max_timestep#max_time_steps_per_repetition


        timesteps    = df['Timesteps'].values.tolist()
        success      = df['Success'].values.tolist()
        feedback     = df['Feedback'].values.tolist()
        pct_feedback = df['Percentage_feedback'].values.tolist()
        e            = df.at[0,'e'].tolist()
        buffer_size  = df.at[0,'Buffer_size'].tolist()
        human_model  = df.at[0,'Human_model'].tolist()
        tau          = df.at[0,'Tau'].tolist()
        abs_pos      = df.at[0, 'Absolute_pos'].tolist()




        # print("timesteps: ", timesteps)
        # print("success: ", success)
        # print("feedback: ", feedback)
        # print("pct_feedback: ", pct_feedback)
        # print("e: ", e)
        # print("buffer_size: ", buffer_size)
        # print("human_model: ", human_model)
        # print("tau: ", tau)



        if human_model == True:
            human_model = 'yes'
        else:
            human_model = 'no'

        if abs_pos == True:
            abs_pos = 'yes'
        else:
            abs_pos = 'no'


        timesteps_list.append(timesteps)
        success_list.append(success)
        feedback_list.append(feedback)
        pct_feedback_list.append(pct_feedback)



    timesteps_list_ok    = []
    success_list_ok      = []
    feedback_list_ok     = []
    pct_feedback_list_ok = []

    for t, s, f, pct_f in zip(timesteps_list, success_list, feedback_list, pct_feedback_list):

        fun_s = interp1d(t, s)
        fun_f = interp1d(t, f)
        fun_pct_f = interp1d(t, pct_f)

        timesteps_last_episode = t[-1]
        timesteps_last_episode = np.int64(timesteps_last_episode)

        timesteps_discretized = np.linspace(0, timesteps_last_episode, num=timesteps_last_episode, endpoint=False)
        timesteps_discretized = timesteps_discretized.astype(int)

        success = fun_s(timesteps_discretized)
        feedback = fun_f(timesteps_discretized)
        pct_feedback = fun_pct_f(timesteps_discretized)



        timesteps_list_ok.append(timesteps_discretized)
        success_list_ok.append(success)
        feedback_list_ok.append(feedback )
        pct_feedback_list_ok.append(pct_feedback)









    return timesteps_list_ok, success_list_ok, feedback_list_ok, pct_feedback_list_ok, tau, e, human_model, buffer_size, abs_pos

