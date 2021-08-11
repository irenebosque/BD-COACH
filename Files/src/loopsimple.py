from post_process_ok import postProcess
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import glob

path = './results/'
repetition_counter = 0



test21ev = 'DCOACH_HM-False_e-0.1_B-10002_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'



tests_Evaluation = [test21ev]

for test_ev in tests_Evaluation:


    test_name_rep_counter = test_ev.replace("{}", "??")

    repetition_counter = 0
    for name in glob.glob(path + test_name_rep_counter):

        print('name: ', name)
        success = []
        feedback = []
        timesteps = []


        for i in range(0, 4):

            name_file = path + test_ev.format(str(i).zfill(2))
            d_frame = pd.read_csv(name_file)
            frame_to_list = d_frame.values.tolist()

            # Timesteps
            del frame_to_list[0][0]
            timesteps.append(frame_to_list[0])
            print('ff',frame_to_list[0])

            # Success this episode: (or return for mountaincar)
            del frame_to_list[14][0]
            success.append(frame_to_list[14])

            # Feedback
            del frame_to_list[2][0]
            feedback.append(frame_to_list[2])



        # Prepare the X axis (the timesteps)
        max_timestep = []
        for item in timesteps:
            max_timestep.append(max(item))
        min_timestep = min(item for item in max_timestep)
        print('min_timestep: ', min_timestep)
        timesteps_ok = list(range(0, int(min_timestep)))


    for f, s in zip(feedback, success):

        feedback_ok = feedback[:int(min_timestep )]
        success_ok = success[:int(min_timestep)]

    for f, s in zip(feedback_ok, success_ok):
        print(s)

        fun_s = interp1d(timesteps_ok, s)
        fun_f = interp1d(timesteps_ok, f)




