from post_process_ok3 import postProcess
import matplotlib.pyplot as plt
import numpy as np




test_HM_0_01_B_500_rand = 'DCOACH_HM-True_e-0.01_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_HM_0_1_B_500_rand = 'DCOACH_HM-True_e-0.1_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'

test_NOHM_0_01_B_500_rand = 'DCOACH_HM-False_e-0.01_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_NOHM_0_1_B_500_rand = 'DCOACH_HM-False_e-0.1_B-500_tau-1e-05_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'





path = './results/'


tests = [test_HM_0_01_B_500_rand, test_HM_0_1_B_500_rand, test_NOHM_0_01_B_500_rand, test_NOHM_0_1_B_500_rand]




cm = 1/2.54

fig, axs= plt.subplots(2, figsize=(17*cm, 17*cm))
ax2 = axs[0].twiny()

ax3 = axs[1].twiny()
ax4= axs[1].twinx()

mujoco_timestep = 0.0125




for test in tests:

    if "Eval-True" in test :
        evaluation = "True"
    else:
        evaluation = "False"

    if "hockey" in test :
        task = "Hockey"
    if "button" in test :
        task = "Button"
    if "reach" in test :
        task = "Reach"
    if "kuka" in test :
        task = "kuka"
    if "rand" in test :
        random = "yes"




    timesteps_processed_list, success_processed_list, feedback_processed_list, pct_feedback_processed_list, \
    tau, e, human_model, buffer_size = postProcess(test, path)


    a = np.array(success_processed_list)
    success_mean = np.mean(a, axis =0)
    success_std = np.std(a, axis=0)

    a = np.array(feedback_processed_list)
    feedback_mean = np.mean(a, axis=0)

    a = np.array(pct_feedback_processed_list)
    pct_feedback_mean = np.mean(a, axis=0)


    a = np.array(timesteps_processed_list)
    timesteps_list = np.mean(a, axis=0)
    simulated_time = timesteps_list * mujoco_timestep /60



    z = np.polyfit(simulated_time, success_mean, 20)

    p = np.poly1d(z)
    print("---------")
    print("p: ", z)
    # ,





    # if human_model == "yes" and e == 0.01:
    #     colorPlot = '#150E56'  # blue
    # if human_model == "yes" and e == 0.1:
    #     colorPlot = '#2978B5'  # green
    # if human_model == "yes" and e == 1:
    #     colorPlot = '#8AB6D6'  # violet
    #
    # if human_model == "no" and e == 0.01:
    #     colorPlot = '#BE0000'  # orange
    # if human_model == "no" and e == 0.1:
    #     colorPlot = '#FF6B6B'  # red
    # if human_model == "no" and e == 1:
    #     colorPlot = '#FDD2BF'  # brown


    # if human_model == "yes" and e == 0.01 and buffer_size == 500:
    #     colorPlot = '#150E56'  # blue
    # if human_model == "yes" and e == 0.1 and buffer_size == 500:
    #     colorPlot = '#2978B5'  # green
    # if human_model == "yes" and e == 1 and buffer_size == 500:
    #     colorPlot = '#8AB6D6'  # violet
    #
    # if human_model == "no" and e == 0.01 and buffer_size == 500:
    #     colorPlot = '#BE0000'  # orange
    # if human_model == "no" and e == 0.1 and buffer_size == 500:
    #     colorPlot = '#FF6B6B'  # red
    # if human_model == "no" and e == 1 and buffer_size == 500:
    #     colorPlot = '#FDD2BF'  # brown

    # if human_model == "no" and e == 0.1 and buffer_size == 500:
    #     colorPlot = '#72956c'  # blue
    # if human_model == "no" and e == 0.1 and buffer_size == 5000:
    #     colorPlot = '#517b5c'  # green
    # if human_model == "no" and e == 0.1 and buffer_size == 10000:
    #     colorPlot = '#2f604b'  # violet
    #
    # if human_model == "yes" and e == 0.1 and buffer_size == 500:
    #     colorPlot = '#ffb600'  # orange
    # if human_model == "yes" and e == 0.1 and buffer_size == 5000:
    #     colorPlot = '#ff7900'  # red
    # if human_model == "yes" and e == 0.1 and buffer_size == 10000:
    #     colorPlot = '#ff4800'  # brown

    if human_model == "yes" and e == 0.1 and buffer_size == 500:
        colorPlot = '#150E56'  # blue
    if human_model == "yes" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#2978B5'  # green
    if human_model == "yes" and e == 0.1 and buffer_size == 10000:
        colorPlot = '#8AB6D6'  # violet

    if human_model == "no" and e == 0.1 and buffer_size == 500:
        colorPlot = '#BE0000'  # orange
    if human_model == "no" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#FF6B6B'  # red
    if human_model == "no" and e == 0.1 and buffer_size == 10000:
        colorPlot = '#FDD2BF'  # brown

    if human_model == "yes" and e == 0.01 and buffer_size == 500:
        colorPlot = '#150E56'  # blue
    if human_model == "yes" and e == 0.01 and buffer_size == 5000:
        colorPlot = '#2978B5'  # green
    if human_model == "yes" and e == 0.01 and buffer_size == 10000:
        colorPlot = '#8AB6D6'  # violet

    if human_model == "no" and e == 0.01 and buffer_size == 500:
        colorPlot = '#BE0000'  # orange
    if human_model == "no" and e == 0.01 and buffer_size == 5000:
        colorPlot = '#FF6B6B'  # red
    if human_model == "no" and e == 0.01 and buffer_size == 10000:
        colorPlot = '#FDD2BF'  # brown








    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)




    axs[0].plot(simulated_time, success_mean, linewidth=0.1, zorder=0, color=colorPlot)
    axs[0].plot(simulated_time, p(simulated_time), linewidth=2.0, color=colorPlot, zorder=1, label='H: ' + human_model + ', Random init: ' + random + ', Buffer size: ' + str(buffer_size) + ', e: ' + str(e))

    axs[0].set_ylabel('% of success')
    axs[0].set_xlabel('min')
    axs[0].set_title('Evaluation of task: Meta-World hockey')
    axs[0].legend(loc='lower right')



    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.set_xlabel('time steps')
    ax2.spines['bottom'].set_position(('outward', 40))
    plt.rc('legend', fontsize=8)  # legend fontsize

    ax2.plot(timesteps_list, p(simulated_time), color=colorPlot, alpha=0)
    plt.xticks(rotation=5)



    # Lower plot:

    axs[1].plot(simulated_time, pct_feedback_mean, linewidth=2.0, color=colorPlot, zorder=0, label='H: ' + human_model + ', Random init: ' + random + ', Buffer size: ' + str(buffer_size) + ', e: ' + str(e))
    axs[1].set_ylabel('% of feedback per episode')
    axs[1].set_xlabel('min')
    axs[1].set_title('Training of task: Meta-World hockey')

    ax3.xaxis.set_ticks_position('bottom')
    ax3.xaxis.set_label_position('bottom')
    ax3.spines['bottom'].set_position(('outward', 40))
    ax3.set_xlabel('time steps')






    ax3.plot(timesteps_list, pct_feedback_mean, color=colorPlot, alpha=0)
    ax4.plot(simulated_time, feedback_mean, color=colorPlot)
    axs[1].legend(loc='upper right')
    plt.xticks(rotation=5)

    ax2.set_xlim([0, 60000])
    ax3.set_xlim([0, 60000])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, 1])






axs[0].grid(linestyle='--')
axs[1].grid(linestyle='--')


#fig.subplots_adjust(hspace=0.2, top=0.95, bottom=0.075) # Space between the subplots
fig.subplots_adjust(hspace=0.4, bottom=0.15,  top=0.96, right=0.6) # Space between the subplots

plt.show()