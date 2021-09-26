from post_process_ok3 import postProcess
import matplotlib.pyplot as plt
import numpy as np

# PUSH

test_HM_0_01_B_15000_rand_push = 'DCOACH_HM-True_e-0.1_B-20000_tau-5e-06_lr-0.005_HMlr-0.001_task-push_rep-rand-init-long-{}.csv'

# HOCKEY
test_HM_0_01_B_15000_rand_hockey = 'DCOACH_HM-True_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_HM_0_01_B_500_rand_hockey = 'DCOACH_HM-True_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_HM_0_1_B_500_rand_hockey = 'DCOACH_HM-True_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'

test_NOHM_0_01_B_15000_rand_hockey = 'DCOACH_HM-False_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_NOHM_0_01_B_500_rand_hockey = 'DCOACH_HM-False_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'
test_NOHM_0_1_B_500_rand_hockey = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-hockey_rep-rand-init-long-{}.csv'

# DOOR





test_HM_0_01_B_15000_rand_door = 'DCOACH_HM-True_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_NOHM_0_01_B_15000_rand_door = 'DCOACH_HM-False_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_HM_0_01_B_500_rand_door = 'DCOACH_HM-True_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_NOHM_0_01_B_500_rand_door = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_HM_0_1_B_15000_rand_door = 'DCOACH_HM-True_e-0.1_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_NOHM_0_1_B_15000_rand_door = 'DCOACH_HM-False_e-0.1_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'

# BUTTON
test_NOHM_0_1_B_500_rand_button = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_NOHM_0_01_B_15000_rand_button = 'DCOACH_HM-False_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_NOHM_0_01_B_500_rand_button = 'DCOACH_HM-False_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'

test_HM_0_1_B_500_rand_button = 'DCOACH_HM-True_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_HM_0_01_B_15000_rand_button = 'DCOACH_HM-True_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_HM_0_01_B_15000_button = 'DCOACH_HM-True_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_HM_0_01_B_500_rand_button = 'DCOACH_HM-True_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_HM_0_1_B_15000_rand_button = 'DCOACH_HM-True_e-0.1_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'
test_HM_0_01_B_45000_rand_button = 'DCOACH_HM-True_e-0.01_B-45000_tau-3e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-rand-init-long-{}.csv'


test_HM_0_1_B_500_button = 'DCOACH_HM-True_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_NOHM_0_01_B_500_button = 'DCOACH_HM-False_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_NOHM_0_1_B_500_button = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_HM_0_01_B_500_button = 'DCOACH_HM-True_e-0.01_B-500_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_NOHM_0_01_B_15000_button = 'DCOACH_HM-False_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
test_HM_0_01_B_15000_button = 'DCOACH_HM-True_e-0.01_B-15000_tau-4e-06_lr-0.005_HMlr-0.001_task-button_topdown_rep-{}.csv'
path = './results/'

#tests = [test_HM_0_01_B_15000_rand_hockey, test_HM_0_01_B_500_rand_hockey, test_HM_0_1_B_500_rand_hockey, test_NOHM_0_01_B_15000_rand_hockey, test_NOHM_0_01_B_500_rand_hockey, test_NOHM_0_1_B_500_rand_hockey]

test_HM_0_01_B_15000_rand_door2 = 'DCOACH_HM-True_e-0.01_B-450001_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm-{}.csv'
test_HM_0_01_B_15000_rand_door3 = 'DCOACH_HM-True_e-0.01_B-450001_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm2-{}.csv'

test_NOHM_0_1_B_500_rand_basketball = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-basketball_rep-randm2-{}.csv'
test_HM_0_01_B_500000_rand_basketball = 'DCOACH_HM-True_e-0.01_B-500000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-basketball_rep-randm2-{}.csv'

#tests = [test_NOHM_0_1_B_15000_rand_door, test_HM_0_1_B_15000_rand_door, test_NOHM_0_01_B_15000_rand_door, test_HM_0_01_B_15000_rand_door]

#tests = [ test_NOHM_0_1_B_500_rand_button, test_HM_0_01_B_15000_button, test_NOHM_0_01_B_15000_rand_button , test_NOHM_0_01_B_500_rand_button, test_HM_0_01_B_500_rand_button, test_HM_0_01_B_15000_rand_button, test_HM_0_01_B_45000_rand_button, test_HM_0_1_B_500_rand_button , test_HM_0_1_B_15000_rand_button]
#tests = [test_HM_0_1_B_500_button, test_HM_0_01_B_500_button, test_HM_0_01_B_15000_button, test_NOHM_0_01_B_500_button, test_NOHM_0_1_B_500_button , test_NOHM_0_01_B_15000_button ]
tests = [test_NOHM_0_1_B_500_rand_basketball, test_HM_0_01_B_500000_rand_basketball]





cm = 1/2.54

fig, axs= plt.subplots(2, figsize=(17*cm, 17*cm))
ax2 = axs[0].twiny()

ax3 = axs[1].twiny()
ax4= axs[1].twinx()

mujoco_timestep = 0.0125

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w




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
    else:
        random = "no"
    if "door" in test :
        task = "door"
    if "push" in test :
        task = "push"
    if "basketball" in test:
        task = "basketball"




    timesteps_processed_list, success_processed_list, feedback_processed_list, pct_feedback_processed_list, \
    tau, e, human_model, buffer_size = postProcess(test, path)


    a = np.array(success_processed_list)
    success_mean = np.mean(a, axis =0)
    success_std = np.std(a, axis=0)

    print('success_mean', success_mean)

    a = np.array(feedback_processed_list)
    feedback_mean = np.mean(a, axis=0)

    a = np.array(pct_feedback_processed_list)
    pct_feedback_mean = np.mean(a, axis=0)


    a = np.array(timesteps_processed_list)
    timesteps_list = np.mean(a, axis=0)
    simulated_time = timesteps_list * mujoco_timestep /60



    z = np.polyfit(simulated_time, success_mean, 5)


    p = np.poly1d(z)
    print("---------")
    print("p: ", z)
    fit_success = p(simulated_time)


    print('success_mean', success_mean)
    fit_success2 = moving_average(success_mean, 5000)
    fit_success2_len = len(fit_success2)
    print("fit_success2_len", fit_success2_len)
    step = simulated_time[-1]/fit_success2_len
    print("step", step)
    print("simulated_time[0]", simulated_time[0])
    print("simulated_time[-1]", simulated_time[-1])

    simulated_time2 = np.arange(simulated_time[0],simulated_time[-1],step)
    print("simulated_time2_len", len(simulated_time2))



    # ,
    # Remove last part of the plots that look a bit weird due to the overfitting of the polynomial.
    # remove_timesteps = 500
    # success_mean = success_mean[:len(success_mean) - remove_timesteps]
    # timesteps_list = timesteps_list[:len(timesteps_list) - remove_timesteps]
    # feedback_mean = feedback_mean[:len(feedback_mean) - remove_timesteps]
    # pct_feedback_mean = pct_feedback_mean[:len(pct_feedback_mean) - remove_timesteps]
    # simulated_time = simulated_time[:len(simulated_time) - remove_timesteps]
    # fit_success = fit_success[:len(fit_success) - remove_timesteps]





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
    # if human_model == "no" and e == 0.1 and buffer_size == 15000:
    #     colorPlot = '#2f604b'  # violet
    #
    # if human_model == "yes" and e == 0.1 and buffer_size == 500:
    #     colorPlot = '#ffb600'  # orange
    # if human_model == "yes" and e == 0.1 and buffer_size == 5000:
    #     colorPlot = '#ff7900'  # red
    # if human_model == "yes" and e == 0.1 and buffer_size == 15000:
    #     colorPlot = '#ff4800'  # brown

    if human_model == "yes" and e == 0.1 and buffer_size == 500:
        colorPlot = '#150E56'  # blue
    if human_model == "yes" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#2978B5'  # green
    if human_model == "yes" and e == 0.1 and buffer_size == 15000:
        colorPlot = '#8AB6D6'  # violet

    if human_model == "no" and e == 0.1 and buffer_size == 500:
        colorPlot = '#BE0000'  # orange
    if human_model == "no" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#FF6B6B'  # red
    if human_model == "no" and e == 0.1 and buffer_size == 15000:
        colorPlot = '#FDD2BF'  # brown



    # e = 0.01, Different buffer sizes
    if human_model == "yes" and e == 0.01 and buffer_size == 500:
        colorPlot = '#150E56'  # blue
    if human_model == "yes" and e == 0.01 and buffer_size == 5000:
        colorPlot = '#2978B5'  # green
    if human_model == "yes" and e == 0.01 and buffer_size == 15000:
        colorPlot = '#8AB6D6'  # violet
    if human_model == "yes" and e == 0.01 and buffer_size == 500000:
        colorPlot = '#0077b6'  # violet

    if human_model == "no" and e == 0.01 and buffer_size == 500:
        colorPlot = '#BE0000'  # orange
    if human_model == "no" and e == 0.01 and buffer_size == 5000:
        colorPlot = '#FF6B6B'  # red
    if human_model == "no" and e == 0.01 and buffer_size == 15000:
        colorPlot = '#FDD2BF'  # brown




    # e = 0.1, Different buffer sizes

    if human_model == "yes" and e == 0.1 and buffer_size == 500:
        colorPlot = '#72956c'  # blue
    if human_model == "yes" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#517b5c'  # green
    if human_model == "yes" and e == 0.1 and buffer_size == 15000:
        colorPlot = '#2f604b'  # violet

    if human_model == "no" and e == 0.1 and buffer_size == 500:
        colorPlot = '#ffb600'  # orange
    if human_model == "no" and e == 0.1 and buffer_size == 5000:
        colorPlot = '#ff7900'  # red
    if human_model == "no" and e == 0.1 and buffer_size == 15000:
        colorPlot = '#ff4800'  # brown












    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)




    axs[0].plot(simulated_time, success_mean, linewidth=0.1, zorder=0, color=colorPlot)
    axs[0].plot(simulated_time, fit_success, linewidth=2.0, zorder=1, color=colorPlot, label='H: ' + human_model + ', Random init: ' + random + ', Buffer size: ' + str(buffer_size) + ', e: ' + str(e))
    axs[0].plot(simulated_time2, fit_success2)
    axs[0].set_ylabel('% of success')
    axs[0].set_xlabel('min')

    title = "Evaluation of task: " + task
    axs[0].set_title(title)
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
    ax4.set_ylabel('Amount of feedback')
    axs[1].set_xlabel('min')
    title = "Evaluation of task: " + task
    axs[1].set_title(title)

    ax3.xaxis.set_ticks_position('bottom')
    ax3.xaxis.set_label_position('bottom')
    ax3.spines['bottom'].set_position(('outward', 40))
    ax3.set_xlabel('time steps')






    ax3.plot(timesteps_list, pct_feedback_mean, color=colorPlot, alpha=0)
    ax4.plot(simulated_time, feedback_mean, color=colorPlot)
    axs[1].legend(loc='lower right')
    plt.xticks(rotation=5)

    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, 1])






axs[0].grid(linestyle='--')
axs[1].grid(linestyle='--')


#fig.subplots_adjust(hspace=0.2, top=0.95, bottom=0.075) # Space between the subplots
fig.subplots_adjust(hspace=0.4, bottom=0.15,  top=0.96, right=0.6) # Space between the subplots

plt.show()