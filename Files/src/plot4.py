from post_process_ok2 import postProcess
import matplotlib.pyplot as plt
import numpy as np


testHM_ev_0_01 = 'DCOACH_HM-True_e-0.01_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
testHM_ev_0_1  = 'DCOACH_HM-True_e-0.1_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
testHM_ev_1    = 'DCOACH_HM-True_e-1.0_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'

testNOHM_ev_0_01 = 'DCOACH_HM-False_e-0.01_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
testNOHM_ev_0_1  = 'DCOACH_HM-False_e-0.1_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
testNOHM_ev_1    = 'DCOACH_HM-False_e-1.0_B-15000_Eval-True_tau-1e-05_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'

testHM_tr_1    = 'DCOACH_HM-True_e-1.0_B-10005_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-00.csv'

#path_ev = './results/'
path_ev = './results/kuka-park/'
path_tr = './results/metaworld-hockey/HM/e_1'



test_kuka_park    = 'DCOACH_HM-True_e-0.3_B-10000_Eval-True_tau-0.0001_lr-0.003_HMlr-0.0003_task-kuka-park-cardboard_rep-{}.csv'
#tests_Evaluation = [testHM_ev_0_01 , testHM_ev_0_1, testHM_ev_1, testNOHM_ev_0_01, testNOHM_ev_0_1, testNOHM_ev_1]
tests_Evaluation = [test_kuka_park]
tests_Training =   [testHM_tr_1]




fig, ax1= plt.subplots(1)



#ax2 = axt.twinx()

ax3 = ax1.twiny()

# for test_tr in tests_Training:
#
#
#     timesteps_processed_list, return_processed_list, feedback_processed_list, tau, e, human_model = postProcess(test_tr, path_tr)
#
# ax3.plot(timesteps_processed_list, feedback_processed_list)

for test_ev in tests_Evaluation:

    if "Eval-True" in test_ev :
        evaluation = "True"
    else:
        evaluation = "False"

    if "hockey" in test_ev :
        task = "Hockey"
    if "button" in test_ev :
        task = "Button"
    if "reach" in test_ev :
        task = "Reach"
    if "kuka" in test_ev :
        task = "kuka"




    timesteps_processed_list, return_processed_list, feedback_processed_list, tau, e, human_model = postProcess(test_ev, path_ev)
    print("return_processed_list: ",return_processed_list)




    # print("human_model: ", human_model)
    # if human_model == "no":
    #     colorPlot = '#1f77b4'  # blue
    # if human_model == "yes":
    #     colorPlot = '#ff7f0e' #orange


    buffer_size = 10000


    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)


    #return_mean = np.average(return_processed_list, axis=0)
    #return_std = np.std(return_processed_list, axis=0)




    ax1.plot(timesteps_processed_list, return_processed_list, linewidth=2.5, zorder=0, label='Human model: ' + human_model + ', B: ' + str(buffer_size) + ', e: ' + str(e) + ', tau: ' + str(tau) + ', task: ' + task + ', Evaluation: ' + evaluation)
    #ax1.fill_between(timesteps_processed_list, return_mean - return_std, return_mean + return_std, alpha=0.1)


    plt.xlabel("Time steps")
    plt.ylim(0, 1.2)
    ax1.set_ylabel('Success rate %')

    ax1.set_title('Task: KUKA park')
    ax1.legend(loc="upper right")




ax1.grid(linestyle='--')

plt.show()