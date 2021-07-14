from post_process_ok import postProcess
import matplotlib.pyplot as plt
import numpy as np

test_button_HM_false_ev_False = 'DCOACH_HM-False_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.007_task-button_rep-{}.csv'
test_button_HM_true_ev_False =  'DCOACH_HM-True_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.007_task-button_rep-{}.csv'

test_reach_HM_false_ev_False =  'DCOACH_HM-False_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.007_task-reach_rep-{}.csv'
test_reach_HM_true_ev_False =  'DCOACH_HM-True_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.007_task-reach_rep-{}.csv'

test_button_HM_false_ev_True =  'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.00016_lr-0.007_task-button_rep-{}.csv'
test_button_HM_true_ev_True =  'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.00016_lr-0.007_task-button_rep-{}.csv'

test_reach_HM_false_ev_True =  'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.00016_lr-0.007_task-reach_rep-07.csv'
test_reach_HM_true_ev_True =  'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.00016_lr-0.007_task-reach_rep-07.csv'


test_hockey_HM_false_ev_True =  'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.001_task-hockey_rep-{}.csv'
test_hockey_HM_true_ev_True =  'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.001_task-hockey_rep-{}.csv'

test_mountaincar_ev_True = 'DCOACH_HM-True_e-0.001_B-1000_Eval-True_tau-0.00045_lr-0.003_task-mountaincar_rep-{}.csv'
test_mountaincar_ev_False = 'DCOACH_HM-True_e-0.001_B-1000_Eval-False_tau-0.00045_lr-0.003_task-mountaincar_rep-{}.csv'

test_vertical_button_HM_true_ev_False = 'DCOACH_HM-True_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.001_task-button_topdpwn_rep-{}.csv'
test_vertical_button_HM_true_ev_True = 'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.001_task-button_topdpwn_rep-{}.csv'

test_vertical_button_HM_false_ev_False = 'DCOACH_HM-False_e-1.0_B-10000_Eval-False_tau-0.00016_lr-0.001_task-button_topdpwn_rep-{}.csv'
test_vertical_button_HM_false_ev_True = 'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.00016_lr-0.001_task-button_topdpwn_rep-{}.csv'


prueba0_005_256 = 'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_task-hockey_tanh_256_rep-{}.csv'
prueba0_005_hm_256 = 'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_task-hockey_tanh_256_rep-{}.csv'
prueba0_005_128 = 'DCOACH_HM-False_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_task-hockey_tanh_128_rep-{}.csv'
prueba0_005_hm_128 = 'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_task-hockey_tanh_128_rep-{}.csv'

prueba0_001 = 'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
prueba0_004 = 'DCOACH_HM-True_e-1.0_B-10000_Eval-True_tau-0.0005_lr-0.005_HMlr-0.0011_task-button_topdown_rep-{}.csv'



tests = [prueba0_001,  prueba0_004]
#tests = [prueba]


fig = plt.figure()
ax1 = fig.add_subplot(111)


ax2 = ax1.twinx()
ax3 = ax1.twiny()

counter_test = 0
for test in tests:

    if "Eval-True" in test:
        evaluation = "True"
    else:
        evaluation = "False"

    if "hockey" in test:
        task = "hockey"

    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model, tau = postProcess(test)

    return_mean = np.average(return_processed_list, axis=0)
    return_std = np.std(return_processed_list, axis=0)
    feedback_mean = np.mean(feedback_processed_list, axis=0)
    t_min_mean = np.mean(t_min_processed_list, axis=0)




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


    task = "vertical button"
    ax1.plot(timesteps_processed_list, return_mean, linewidth=2.5,  label='Human model: ' + human_model + ', B: ' + str(buffer_size) + ', e: ' + str(e) + ', tau: ' + str(tau) + ', task: ' + task + ', Evaluation: ' + evaluation)

    ax1.fill_between(range(min_index), return_mean - return_std, return_mean + return_std, alpha = 0.1)



    plt.xlabel("Time steps")
    #plt.ylim(0, 0.02)
    ax1.set_ylabel('Success rate %')
    ax2.set_ylabel('% of feedback given by the oracle')
    ax1.set_title('Task')

    ax2.plot(timesteps_processed_list, feedback_mean, '--',color='gray', linewidth=1)


    ax3.plot(t_min_mean, feedback_mean, color='white', linewidth=0.1)
    ax3.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax3.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax3.spines['bottom'].set_position(('outward', 28))
    ax3.set_xlabel('Time (min)')

    ax1.legend(loc="upper right")
    counter_test += 1



ax1.grid(linestyle='--')

plt.show()