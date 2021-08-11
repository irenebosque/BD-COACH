from post_process_ok import postProcess
import matplotlib.pyplot as plt
import numpy as np




test3 = 'DCOACH_HM-True_e-0.01_B-3000_Eval-True_tau-0.0003_lr-0.005_HMlr-0.005_task-hockey_rep-{}.csv'
test4 = 'DCOACH_HM-True_e-0.01_B-3000_Eval-True_tau-0.000301_lr-0.005_HMlr-0.005_task-hockey_rep-{}.csv'
test5 = 'DCOACH_HM-True_e-0.01_B-3000_Eval-True_tau-0.0003011_lr-0.005_HMlr-0.005_task-hockey_rep-{}.csv' # HM comesinto action after 150 h signals
test6 = 'DCOACH_HM-True_e-0.01_B-3000_Eval-True_tau-0.0003011_lr-0.005_HMlr-0.001_task-hockey_rep-00.csv'
test7 = 'DCOACH_HM-False_e-0.1_B-3000_Eval-True_tau-0.0003011_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test8 = 'DCOACH_HM-True_e-0.1_B-3000_Eval-True_tau-0.0003011_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test9 = 'DCOACH_HM-False_e-0.1_B-3000_Eval-True_tau-0.0003012_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test10 = 'DCOACH_HM-True_e-0.1_B-3000_Eval-True_tau-0.0003012_lr-0.005_HMlr-0.001_task-hockey_rep-{}.csv'
test11 = 'DCOACH_HM-True_e-0.1_B-3000_Eval-False_tau-0.0003012_lr-0.005_HMlr-0.001_task-hockey_rep-12.csv'
test11 = 'DCOACH_HM-True_e-0.1_B-3000_Eval-False_tau-0.0003012_lr-0.005_HMlr-0.001_task-hockey_rep-12.csv'

test12 = 'DCOACH_HM-True_e-0.1_B-3000_Eval-False_tau-0.0003013_lr-0.005_HMlr-0.001_task-hockey_rep-00.csv'
test12ev = 'DCOACH_HM-True_e-0.1_B-3000_Eval-True_tau-0.0003013_lr-0.005_HMlr-0.001_task-hockey_rep-00.csv'
#tests = [test1, test2, test3, test4, test5, test6]
#tests = [test9, test10]
tests_Evaluation = [test12ev]
tests_Train = [test12]

#tests = [prueba]


#fig = plt.figure()
#ax1 = fig.add_subplot(111)


fig, (ax1, axt)  = plt.subplots(2)


ax2 = ax1.twinx()
ax4 = ax1.twinx()
ax3 = ax1.twiny()

counter_test = 0
for test_ev in tests_Evaluation:

    if "Eval-True" in test:
        evaluation = "True"
    else:
        evaluation = "False"

    if "hockey" in test:
        task = "Hockey"
    if "button" in test:
        task = "Button"
    if "reach" in test:
        task = "Reach"


    #timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau = postProcess(test)
    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau, pl_ag_processed_list, pl_hm_processed_list = postProcess(test_ev)
    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau, pl_ag_processed_list, pl_hm_processed_list = postProcess(
        test_tr)


    return_mean = np.average(return_processed_list, axis=0)
    return_std = np.std(return_processed_list, axis=0)
    feedback_mean = np.mean(feedback_processed_list, axis=0)
    t_min_mean = np.mean(t_min_processed_list, axis=0)

    # NEW ####
    pl_ag_mean = np.average(pl_ag_processed_list, axis=0)
    pl_hm_mean = np.average(pl_hm_processed_list, axis=0)
    ##########




    counter = 0

    for i in return_processed_list:
        if counter_test  == 0:
            colorPlot = '#1f77b4' #blue
        if counter_test  == 1:
            colorPlot = '#2ca02c' #green
        if counter_test  == 2:
            colorPlot = '#c875C4' #violet
        if counter_test  == 3:
            colorPlot = '#ff7f0e' #orange
        if counter_test  == 4:
            colorPlot = '#d62728' #red
        if counter_test  == 5:
            colorPlot = '#A9561E' #brown

        #ax1.plot(timesteps_processed_list, return_processed_list[counter],'--', color = colorPlot , linewidth=1)
        counter +=1

    linestyle_plot = 'dotted'

    if tau == 0.0002:
        linestyle_plot = 'solid'
    if tau == 0.0003:
        linestyle_plot = 'dashed'
    if tau == 0.00045:
        linestyle_plot = 'dashdot'
    if tau == 0.0007:
        linestyle_plot = 'dotted'

    if e == 0.001:
        colorPlot = '#1f77b4'  # blue
    if e == 0.01:
        colorPlot = '#ff7f0e' #orange
    if e == 0.1:
        colorPlot = 'green'  # blue
    if e == 1:
        colorPlot = 'red' #orange


    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)



    ax1.plot(timesteps_processed_list, return_mean, linewidth=2.5,  label='Human model: ' + human_model + ', B: ' + str(buffer_size) + ', e: ' + str(e) + ', tau: ' + str(tau) + ', task: ' + task + ', Evaluation: ' + evaluation)

    ax1.fill_between(range(min_index), return_mean - return_std, return_mean + return_std, alpha = 0.1)



    plt.xlabel("Time steps")
    #plt.ylim(0, 0.02)
    ax1.set_ylabel('Success rate %')
    ax2.set_ylabel('% of feedback given by the oracle')
    ax1.set_title('Task')

    ax2.plot(timesteps_processed_list, feedback_mean, '--',color='gray', linewidth=1)

    ax4.plot(timesteps_processed_list, pl_ag_mean, '--', color='gray', linewidth=1)
    # NEW ####
    ax4.plot(timesteps_processed_list, pl_ag_mean, '--', color='gray', linewidth=1)
    ##########


    ax3.plot(t_min_mean, feedback_mean, color='white', linewidth=0.1)
    ax3.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax3.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax3.spines['bottom'].set_position(('outward', 28))
    ax3.set_xlabel('Time (min)')

    ax1.legend(loc="upper right")
    counter_test += 1



ax1.grid(linestyle='--')

plt.show()