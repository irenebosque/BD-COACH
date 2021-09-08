from post_process_ok import postProcess
import matplotlib.pyplot as plt
import numpy as np



test12   = 'DCOACH_HM-True_e-0.1_B-3004_Eval-False_tau-0.0003_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test12ev = 'DCOACH_HM-True_e-0.1_B-3004_Eval-True_tau-0.0003_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'

test13   = 'DCOACH_HM-True_e-0.1_B-3000_Eval-False_tau-0.0003_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test13ev = 'DCOACH_HM-True_e-0.1_B-3000_Eval-True_tau-0.0003_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'

test14   = 'DCOACH_HM-False_e-0.1_B-10000_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test14ev = 'DCOACH_HM-False_e-0.1_B-10000_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'

test15   = 'DCOACH_HM-True_e-0.1_B-10000_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test15ev = 'DCOACH_HM-True_e-0.1_B-10000_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'



test16   = 'DCOACH_HM-True_e-0.1_B-10000_Eval-False_tau-0.0001_lr-0.005_HMlr-0.003_task-hockey_rep-{}.csv'
test16ev = 'DCOACH_HM-True_e-0.1_B-10000_Eval-True_tau-0.0001_lr-0.005_HMlr-0.003_task-hockey_rep-{}.csv'

test17   = 'DCOACH_HM-True_e-0.1_B-10000_Eval-False_tau-0.0001_lr-0.005_HMlr-0.005_task-hockey_rep-{}.csv'
test17ev = 'DCOACH_HM-True_e-0.1_B-10000_Eval-True_tau-0.0001_lr-0.005_HMlr-0.005_task-hockey_rep-{}.csv'

test18   = 'DCOACH_HM-True_e-0.1_B-10001_Eval-False_tau-0.0001_lr-0.005_HMlr-0.003_task-hockey_rep-{}.csv'
test18ev = 'DCOACH_HM-True_e-0.1_B-10001_Eval-True_tau-0.0001_lr-0.005_HMlr-0.003_task-hockey_rep-{}.csv'

test19   = 'DCOACH_HM-True_e-0.1_B-10001_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test19ev = 'DCOACH_HM-True_e-0.1_B-10001_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'

test20   = 'DCOACH_HM-True_e-0.1_B-10002_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test20ev = 'DCOACH_HM-True_e-0.1_B-10002_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'


test21   = 'DCOACH_HM-False_e-0.1_B-10002_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test21ev = 'DCOACH_HM-False_e-0.1_B-10003_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-00.csv'


test00   = 'DCOACH_HM-False_e-0.1_B-10004_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test00ev = 'DCOACH_HM-False_e-0.1_B-10004_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'


test01   = 'DCOACH_HM-True_e-0.1_B-10004_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test01ev = 'DCOACH_HM-True_e-0.1_B-10004_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'



test02 =   'DCOACH_HM-True_e-0.1_B-10005_Eval-False_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test02ev = 'DCOACH_HM-True_e-0.1_B-10006_Eval-True_tau-0.0001_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'

tests_Train = [test02]
tests_Evaluation = [test02ev]




fig, (ax1, ax2, axt2) = plt.subplots(3)



#ax2 = axt.twinx()

ax3 = ax1.twiny()


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


    #timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau = postProcess(test)
    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau, pl_ag_processed_list, pl_hm_processed_list = postProcess(test_ev)
    # timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau, pl_ag_processed_list, pl_hm_processed_list = postProcess(
    #     test_tr)
    #
    #print("return_processed_list: ", return_processed_list)

    return_mean = np.average(return_processed_list, axis=0)
    return_std = np.std(return_processed_list, axis=0)
    feedback_mean = np.mean(feedback_processed_list, axis=0)
    t_min_mean = np.mean(t_min_processed_list, axis=0)


    z = np.polyfit(timesteps_processed_list, return_mean, 3)

    p = np.poly1d(z)



    print("human_model: ", human_model)
    if human_model == "no":
        colorPlot = '#1f77b4'  # blue
    if human_model == "yes":
        colorPlot = '#ff7f0e' #orange




    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)



    ax1.plot(timesteps_processed_list, return_mean, linewidth=2.5, zorder=0, color=colorPlot, label='Human model: ' + human_model + ', B: ' + str(buffer_size) + ', e: ' + str(e) + ', tau: ' + str(tau) + ', task: ' + task + ', Evaluation: ' + evaluation)

    ax1.fill_between(range(min_index), return_mean - return_std, return_mean + return_std, alpha = 0.1)
    #ax1.plot(timesteps_processed_list, p(timesteps_processed_list) , linewidth=5, color = colorPlot, zorder=3)


    plt.xlabel("Time steps")
    #plt.ylim(0, 0.02)
    ax1.set_ylabel('Success rate %')

    ax1.set_title('Task: KUKA park')
    ax1.legend(loc="upper right")






    # ax3.plot(t_min_mean, feedback_mean, color='white', linewidth=0.1)
    # ax3.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    # ax3.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    # ax3.spines['bottom'].set_position(('outward', 28))
    # ax3.set_xlabel('Time (min)')

for test_tr in tests_Train:

    if "Eval-True" in test_tr :
        evaluation = "True"
    else:
        evaluation = "False"

    if "hockey" in test_tr :
        task = "Hockey"
    if "button" in test_tr :
        task = "Button"
    if "reach" in test_tr :
        task = "Reach"


    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau, pl_ag_processed_list, pl_hm_processed_list = postProcess(test_tr)

    feedback_mean = np.mean(feedback_processed_list, axis=0)

    ax2.plot(timesteps_processed_list, feedback_mean)

    t_min_mean = np.mean(t_min_processed_list, axis=0)

    # NEW ####
    pl_ag_mean = np.average(pl_ag_processed_list, axis=0)
    pl_hm_mean = np.average(pl_hm_processed_list, axis=0)
    pl_ag_std = np.average(pl_ag_processed_list, axis=0)
    pl_hm_std = np.std(pl_hm_processed_list, axis=0)

    feedback_std = np.average(feedback_processed_list, axis=0)

    # ##########
    # axf.set_ylabel('% of feedback given by the oracle')
    # axf.plot(timesteps_processed_list, feedback_mean, '--', linewidth=1)
    # axf.fill_between(range(min_index), feedback_mean - feedback_std, feedback_mean + feedback_std, alpha=0.1)

    #axt2 = axt.twinx()
    # axt.plot(timesteps_processed_list, pl_ag_mean, '--',color='red', linewidth=1)
    # axt.fill_between(range(min_index), pl_ag_mean - pl_ag_std, pl_ag_mean + pl_ag_std, color='red',alpha=0.1)

    # NEW ####
    axt2.plot(timesteps_processed_list, pl_hm_mean, '--', linewidth=1)
    axt2.fill_between(range(min_index), pl_hm_mean - pl_hm_std, pl_hm_mean + pl_hm_std, alpha=0.1)
    ##########





ax1.grid(linestyle='--')
ax2.grid(linestyle='--')
axt2.grid(linestyle='--')
#axf.grid(linestyle='--')
#axt.grid(linestyle='--')
plt.show()