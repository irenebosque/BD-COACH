
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from collections import Counter
import glob

testHM_0_1_B_500_rand =  'DCOACH_HM-True_e-0.1_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
testHM_0_1_B_5000_rand =  'DCOACH_HM-True_e-0.1_B-5000_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
testHM_0_1_B_10000_rand = 'DCOACH_HM-True_e-0.1_B-10000_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'

testNOHM_0_1_B_500_rand =  'DCOACH_HM-False_e-0.1_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
testNOHM_0_1_B_5000_rand =  'DCOACH_HM-False_e-0.1_B-5000_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
testNOHM_0_1_B_10000_rand =  'DCOACH_HM-False_e-0.1_B-10000_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'


path = './results/'
tests = [testHM_0_1_B_500_rand, testHM_0_1_B_5000_rand, testHM_0_1_B_10000_rand, testNOHM_0_1_B_500_rand, testNOHM_0_1_B_5000_rand, testNOHM_0_1_B_10000_rand]

for test in tests:


    repetition_counter = 0

    test_name_rep_counter = test.replace("{}", "??")

    for name in glob.glob(path + test_name_rep_counter):
        repetition_counter += 1
        print('name: ', name)
    print('repetition_counter', repetition_counter)


    for i in range(0, repetition_counter):
        name_test = path + test.format(str(i).zfill(2))

        df = pd.read_csv(name_test)

        df.iloc[1 , -1] = 60500
        print(df)

        df.to_csv("name_test.csv", index=False)







