from post_process_ok import postProcess
import matplotlib.pyplot as plt
import numpy as np
print('OLAAA')
### Human model included (hm)
#test0_hm = 'DCOACH_yes_Human_model_e-0.001_buffer-10000_repetition-{}_tau-0.00045_Evaluation-False.csv'
test1_hm = 'DCOACH_yes_Human_model_e-0.001_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test2_hm = 'DCOACH_yes_Human_model_e-0.01_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test3_hm = 'DCOACH_yes_Human_model_e-1_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test4_hm = 'DCOACH_yes_Human_model_e-1_buffer-100_repetition-{}_tau-0.00045_Evaluation-False.csv'
test5_hm = 'DCOACH_yes_Human_model_e-1_buffer-10000_repetition-{}_tau-0.00045_Evaluation-False.csv'


### Human model NOT included (-)
#test0 = 'DCOACH_no_Human_model_e-0.001_buffer-10000_repetition-{}_tau-0.00045_Evaluation-False.csv'
test1 = 'DCOACH_no_Human_model_e-0.001_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test2 = 'DCOACH_no_Human_model_e-0.01_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test3 = 'DCOACH_no_Human_model_e-1_buffer-1000_repetition-{}_tau-0.00045_Evaluation-True.csv'
test4 = 'DCOACH_no_Human_model_e-1_buffer-100_repetition-{}_tau-0.00045_Evaluation-False.csv'
#test5 = 'DCOACH_no_Human_model_e-1_buffer-10000_repetition-{}_tau-0.00045_Evaluation-False.csv'

test9 = 'DCOACH_yes_Human_model_e-0.001_buffer-1000_tau-0.00045_Evaluation-True-rep-{}.csv'


test1_hm = 'DCOACH_Human_model_included-True_e-0.001_buffer-1000_Evaluation-False_repetition-{}.csv'
test2_hm = 'DCOACH_Human_model_included-True_e-0.01_buffer-1000_Evaluation-False_repetition-{}.csv'
test3_hm = 'DCOACH_Human_model_included-True_e-0.1_buffer-1000_Evaluation-False_repetition-{}.csv'
test4_hm = 'DCOACH_Human_model_included-True_e-1_buffer-1000_Evaluation-False_repetition-{}.csv'


test1 = 'DCOACH_Human_model_included-False_e-0.001_buffer-1000_Evaluation-False_repetition-{}.csv'
test2 = 'DCOACH_Human_model_included-False_e-0.01_buffer-1000_Evaluation-False_repetition-{}.csv'
test3 = 'DCOACH_Human_model_included-False_e-0.1_buffer-1000_Evaluation-False_repetition-{}.csv'
test4 = 'DCOACH_Human_model_included-False_e-1_buffer-1000_Evaluation-False_repetition-{}.csv'

test = 'DCOACH_Human_model_included-False_e-0.01_buffer-1000_Evaluation-False_repetition-{}.csv'

test1_ev = 'DCOACH_Human_model_included-False_e-0.001_buffer-1000_Evaluation-True_repetition-{}.csv'
test2_ev = 'DCOACH_Human_model_included-False_e-0.01_buffer-1000_Evaluation-True_repetition-{}.csv'
test3_ev = 'DCOACH_Human_model_included-False_e-0.1_buffer-1000_Evaluation-True_repetition-{}.csv'
test4_ev = 'DCOACH_Human_model_included-False_e-1_buffer-1000_Evaluation-True_repetition-{}.csv'

test1_hm_ev = 'DCOACH_Human_model_included-True_e-0.001_buffer-1000_Evaluation-True_repetition-{}.csv'
test2_hm_ev = 'DCOACH_Human_model_included-True_e-0.01_buffer-1000_Evaluation-True_repetition-{}.csv'
test3_hm_ev = 'DCOACH_Human_model_included-True_e-0.1_buffer-1000_Evaluation-True_repetition-{}.csv'
test4_hm_ev = 'DCOACH_Human_model_included-True_e-1_buffer-1000_Evaluation-True_repetition-{}.csv'

tests = [test1_hm, test2_hm, test3_hm, test4_hm, test1, test2, test3, test4]
tests = [test3_hm, test3]
#tests = [test1_hm_ev, test2_hm_ev, test3_hm_ev, test4_hm_ev, test1_ev, test2_ev, test3_ev, test4_ev]
#tests = [test1, test1_ev, test1_hm, test1_hm_ev]
#tests = [ test1, test2,test3, test4]


fig = plt.figure()
ax1 = fig.add_subplot(111)


ax2 = ax1.twinx()
ax3 = ax1.twiny()

counter_test = 0
for test in tests:

    timesteps_processed_list, return_processed_list, feedback_processed_list, t_min_processed_list, min_index, e, buffer_size, human_model = postProcess(test)

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




    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)



    ax1.plot(timesteps_processed_list, return_mean, linewidth=2.5, label='Return --> ' + 'Human model: ' + human_model + ', B: ' + str(buffer_size) + ', e: ' + str(e))

    ax1.fill_between(range(min_index), return_mean - return_std, return_mean + return_std, alpha = 0.2)



    plt.xlabel("Time steps")
    ax1.set_ylabel('Return')
    ax2.set_ylabel('% of feedback given by the teacher')
    ax1.set_title('MountainCar-v0 (low dimension) Learning Curve')

    ax2.plot(timesteps_processed_list, feedback_mean, '--',color='gray', linewidth=1)


    ax3.plot(t_min_mean, feedback_mean, color='white', linewidth=0.1)
    ax3.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax3.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax3.spines['bottom'].set_position(('outward', 28))
    ax3.set_xlabel('Time (min)')

    ax1.legend()
    counter_test += 1



ax1.grid(linestyle='--')

plt.show()