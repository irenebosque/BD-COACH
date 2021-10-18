from post_process_ok3 import postProcess
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal

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
test_HM_0_01_B_15000_rand_door3 = 'DCOACH_HM-True_e-0.01_B-50000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-door_rep-randm2-{}.csv'

test_NOHM_0_1_B_500_rand_basketball = 'DCOACH_HM-False_e-0.1_B-500_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-basketball_rep-randm2-{}.csv'
test_HM_0_01_B_500000_rand_basketball = 'DCOACH_HM-True_e-0.01_B-500000_tau-4e-06_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-basketball_rep-randm2-{}.csv'
test_HM_0_1_B_500000_rand_soccer = 'DCOACH_HM-True_e-0.1_B-500000_tau-3e-06_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-{}.csv'
test_NOHM_0_1_B_500_rand_soccer = 'DCOACH_HM-False_e-0.1_B-500_tau-3e-06_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-{}.csv'
test_NOHM_0_1_B_500_rand_soccer_long = 'DCOACH_HM-False_e-0.1_B-500_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_4m-{}.csv'
test_HM_0_1_B_500000_rand_soccer_long = 'DCOACH_HM-True_e-0.1_B-500000_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_4m-{}.csv'
test_NOHM_0_1_B_500_rand_soccer_short = 'DCOACH_HM-False_e-0.1_B-500_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_v2-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short = 'DCOACH_HM-True_e-0.1_B-500000_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_v2-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_obs = 'DCOACH_HM-True_e-0.1_B-500000_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_obs-{}.csv'
test_NOHM_0_1_B_500_rand_soccer_short = 'DCOACH_HM-False_e-0.1_B-500_tau-5e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_v2-{}.csv'




test_NOHM_0_1_B_500_rand_soccer_short_org_obs = 'DCOACH_HM-False_e-0.1_B-500_tau-1e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_01 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.01_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_v3 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_v3-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.005_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-big-net-{}.csv'


test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B30 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.005_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling30-{}.csv'

test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentHMlr_0_001_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.001_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentHMlr_0_01_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.005_HMlr-0.001_agent_batch_lr-0.01_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'

test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_003_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.003_HMlr-0.001_agent_batch_lr-0.001_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'
test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_001_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.001_HMlr-0.001_agent_batch_lr-0.001_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'

test_NOHM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_001_bignet_B20 = 'DCOACH_HM-False_e-0.1_B-500_tau-1e-07_lr-0.001_HMlr-0.001_agent_batch_lr-0.001_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'


test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_001_HM0_0005_bignet_B20 = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.001_HMlr-0.0005_agent_batch_lr-0.001_task-soccer_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'

test_HM_0_1_B_500000_rand_basket_abs_pos = 'DCOACH_HM-True_e-0.1_B-500000_tau-1e-07_lr-0.001_HMlr-0.001_agent_batch_lr-0.001_task-basketball_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'
test_NOHM_0_1_B_500_rand_basket_abs_pos = 'DCOACH_HM-False_e-0.1_B-500_tau-1e-07_lr-0.001_HMlr-0.001_agent_batch_lr-0.001_task-basketball_rep-randm-0_2m_org_obs-big-net-B-sampling20-{}.csv'




tests = [test_NOHM_0_1_B_500_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_01,test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet, test_HM_0_1_B_500000_rand_soccer_short_v3, test_HM_0_1_B_500000_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B20, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B30]
tests = [test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B20, test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentHMlr_0_001_bignet_B20, test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentHMlr_0_01_bignet_B20]

tests = [test_NOHM_0_1_B_500_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_v3, test_HM_0_1_B_500000_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_003_bignet_B20]#test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_01,test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet, test_HM_0_1_B_500000_rand_soccer_short_v3, test_HM_0_1_B_500000_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B20, test_HM_0_1_B_500000_rand_soccer_short_org_obs_HMlr_0_001_bignet_B30]
tests = [test_HM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_001_bignet_B20,test_NOHM_0_1_B_500000_rand_soccer_short_org_obs_agentlr_0_001_bignet_B20, test_NOHM_0_1_B_500_rand_soccer_short_org_obs, test_HM_0_1_B_500000_rand_soccer_short_v3]

tests = [test_HM_0_1_B_500000_rand_basket_abs_pos, test_NOHM_0_1_B_500_rand_basket_abs_pos]

test_NOHM_e_0_01_B_500_button_rel = 'HM-False_e-0.01_B-500_task-button_topdown_rep-{}.csv'
test_NOHM_e_0_1_B_500_button_rel = 'HM-False_e-0.1_B-500_task-button_topdown_rep-{}.csv'
test_NOHM_e_1_B_500_button_rel = 'HM-False_e-1.0_B-500_task-button_topdown_rep-{}.csv'

test_HM_e_0_01_B_500000_button_rel = 'HM-True_e-0.01_B-500000_task-button_topdown_rep-{}.csv'
test_HM_e_0_1_B_500000_button_rel = 'HM-True_e-0.1_B-500000_task-button_topdown_rep-{}.csv'
test_HM_e_1_B_500000_button_rel = 'HM-True_e-1.0_B-500000_task-button_topdown_rep-{}.csv'

test_HM_e_0_01_B_500000_button_abs = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-{}.csv'
test_HM_e_1_B_500000_button_abs = 'HM-True_e-1.0_B-500000_task-button_topdown_absolute_pos-True_rep-{}.csv'
test_NOHM_e_1_B_500_button_abs = 'HM-False_e-1.0_B-500_task-button_topdown_absolute_pos-True_rep-{}.csv'
test_HM_e_1_B_500000_button_absv2 = 'HM-True_e-1.0_B-500000_task-button_topdown_absolute_pos-True_repv2-{}.csv' #bigger net
test_HM_e_0_1_B_500000_button_absv2 = 'HM-True_e-0.1_B-500000_task-button_topdown_absolute_pos-True_repv2-{}.csv'
test_HM_e_0_01_B_500000_button_absv2 = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_repv2-{}.csv'
# tests = [test_NOHM_e_0_01_B_500_button_rel,
#          test_NOHM_e_0_1_B_500_button_rel,
#          test_NOHM_e_1_B_500_button_rel,
#          test_HM_e_0_01_B_500000_button_rel,
#          test_HM_e_0_1_B_500000_button_rel,
#          test_HM_e_1_B_500000_button_rel,
#          test_HM_e_0_01_B_500000_button_abs,
#          test_NOHM_e_1_B_500_button_abs]

tests = [test_NOHM_e_1_B_500_button_rel,
         test_NOHM_e_1_B_500_button_abs,
         test_HM_e_0_01_B_500000_button_rel,
         test_HM_e_0_01_B_500000_button_abs,
         test_HM_e_1_B_500000_button_abs,
         test_HM_e_1_B_500000_button_absv2,
         test_HM_e_0_1_B_500000_button_absv2,
         test_HM_e_0_01_B_500000_button_absv2]

test_HM_e_0_1_B_500000_button_abs = 'HM-True_e-0.01_B-500000_task-door_absolute_pos-True_rep-{}.csv'
test_NOHM_e_0_1_B_500_button_abs = 'HM-False_e-0.1_B-500_task-door_absolute_pos-True_rep-{}.csv'

test_NOHM_e_0_1_B_500_basket_abs = 'HM-False_e-0.1_B-500_task-basketball_absolute_pos-True_rep-{}.csv'
test_HM_e_0_1_B_500000_basket_abs = 'HM-True_e-0.1_B-500000_task-basketball_absolute_pos-True_rep-{}.csv'

tests = [test_HM_e_0_1_B_500000_button_abs, test_NOHM_e_0_1_B_500_button_abs ]
tests = [test_NOHM_e_0_1_B_500_basket_abs, test_HM_e_0_1_B_500000_basket_abs ]


test_NOHM_e_0_1_B_500_reach_abs = 'HM-False_e-0.1_B-500_task-reach_absolute_pos-True_rep-{}.csv'
test_HM_e_0_01_B_500000_reach_abs = 'HM-True_e-0.01_B-500000_task-reach_absolute_pos-True_rep-v2-{}.csv'
test_HM_e_0_01_B_500000_reach_abs_v3 = 'HM-True_e-0.01_B-500000_task-reach_absolute_pos-True_rep-v3-{}.csv' #smaller HM lr and theta = 0.05


tests = [test_NOHM_e_0_1_B_500_reach_abs, test_HM_e_0_01_B_500000_reach_abs, test_HM_e_0_01_B_500000_reach_abs_v3]

test_HM_e_0_01_B_500000_soccer_abs = 'HM-True_e-0.01_B-500000_task-soccer_absolute_pos-True_rep-v3-{}.csv'
test_NOHM_e_0_1_B_500_soccer_abs = 'HM-False_e-0.1_B-500_task-soccer_absolute_pos-True_rep-v3-{}.csv'
test_HM_e_0_01_B_500000_soccer_abs_v4 = 'HM-True_e-0.01_B-500000_task-soccer_absolute_pos-True_rep-v4-{}.csv' # tanh normal y 512 en vez de 128 (50000 t)
tests = [test_NOHM_e_0_1_B_500_soccer_abs, test_HM_e_0_01_B_500000_soccer_abs, test_HM_e_0_01_B_500000_soccer_abs_v4]

test_NOHM_e_0_1_B_500_soccer_abs_shorter = 'HM-False_e-0.1_B-500_task-soccer_absolute_pos-True_rep-v4-{}.csv' #same as previous without hm but 50000 t
tests = [test_NOHM_e_0_1_B_500_soccer_abs_shorter, test_HM_e_0_01_B_500000_soccer_abs_v4]
#
# test_NOHM_e_0_1_B_500_sweep_abs_shorter = 'HM-False_e-0.1_B-500_task-sweep_absolute_pos-True_rep-v5-{}.csv'
# test_HM_e_0_1_B_500_sweep_abs_shorter = 'HM-True_e-0.01_B-500000_task-sweep_absolute_pos-True_rep-v5-{}.csv'
# tests = [test_NOHM_e_0_1_B_500_sweep_abs_shorter, test_HM_e_0_1_B_500_sweep_abs_shorter]
#
test_NOHM_e_0_1_B_500_basket_abs_shorter = 'HM-False_e-0.1_B-500_task-basketball_absolute_pos-True_rep-v5-{}.csv'
test_HM_e_0_1_B_500_basket_abs_shorter = 'HM-True_e-0.01_B-500000_task-basketball_absolute_pos-True_rep-v5-{}.csv'
tests = [test_NOHM_e_0_1_B_500_basket_abs_shorter, test_HM_e_0_1_B_500_basket_abs_shorter]

# test_NOHM_e_0_1_B_500_door_abs_shorter = 'HM-False_e-0.1_B-500_task-door_absolute_pos-True_rep-v5-{}.csv'
# test_HM_e_0_1_B_500_door_abs_shorter = 'HM-True_e-0.01_B-500000_task-door_absolute_pos-True_rep-v5-{}.csv'
# tests = [test_NOHM_e_0_1_B_500_door_abs_shorter, test_HM_e_0_1_B_500_door_abs_shorter]
#
test_NOHM_e_0_1_B_500_hockey_abs_shorter = 'HM-False_e-0.1_B-500_task-hockey_absolute_pos-True_rep-v5-{}.csv'
test_HM_e_0_1_B_50000_hockey_abs_shorter = 'HM-True_e-0.01_B-500000_task-hockey_absolute_pos-True_rep-v5-{}.csv'
tests = [test_NOHM_e_0_1_B_500_hockey_abs_shorter, test_HM_e_0_1_B_50000_hockey_abs_shorter]


# test_NOHM_e_0_1_B_500_drawer_abs_shorter = 'HM-False_e-0.1_B-500_task-drawer_absolute_pos-True_rep-v5-{}.csv'
# test_HM_e_0_1_B_500_drawer_abs_shorter = 'HM-True_e-0.01_B-500000_task-drawer_absolute_pos-True_rep-v5-{}.csv'
# tests = [test_NOHM_e_0_1_B_500_drawer_abs_shorter, test_HM_e_0_1_B_500_drawer_abs_shorter]

test_NOHM_e_0_1_B_500_button_topdown_abs_shorter = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-v5-{}.csv'
test_HM_e_0_1_B_500_button_topdown_abs_shorter = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-v5-{}.csv'
tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_shorter, test_HM_e_0_1_B_500_button_topdown_abs_shorter]

test_NOHM_e_0_1_B_500_button_topdown_abs_shorter = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-v6-{}.csv'
test_HM_e_0_1_B_50000_button_topdown_abs_shorter = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-v6-{}.csv'
tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_shorter, test_HM_e_0_1_B_50000_button_topdown_abs_shorter]

test_NOHM_e_0_1_B_500_button_topdown_abs_shorter = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-v7-{}.csv'
test_HM_e_0_1_B_50000_button_topdown_abs_shorter = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-v7-{}.csv'
tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_shorter, test_HM_e_0_1_B_50000_button_topdown_abs_shorter]

test_NOHM_e_0_1_B_500_button_topdown_abs_shorter_alpha06 = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-alpha06-{}.csv'
test_HM_e_0_1_B_50000_button_topdown_abs_shorter_alpha06 = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-alpha06-{}.csv'
tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_shorter_alpha06, test_HM_e_0_1_B_50000_button_topdown_abs_shorter_alpha06]



test_HM_e_0_1_B_50000_button_topdown_abs_shorter_alpha07 = 'HM-True_e-0.01_B-500000_task-button_topdown_absolute_pos-True_rep-alpha07-{}.csv'
test_NOHM_e_0_1_B_500_button_topdown_abs_shorter_alpha07 = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-alpha07-{}.csv'
tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_shorter_alpha07, test_HM_e_0_1_B_50000_button_topdown_abs_shorter_alpha07]


test_HM_e_0_01_B_50000_button_topdown_abs_alpha07v2 = 'HM-True_e-0.01_B-50000_task-button_topdown_absolute_pos-True_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_500_button_topdown_abs_alpha07v2 = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-True_rep-alpha07v2-{}.csv'
test_HM_e_0_01_B_50000_button_topdown_rel_alpha07v2 = 'HM-True_e-0.01_B-50000_task-button_topdown_absolute_pos-False_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_500_button_topdown_rel_alpha07v2 = 'HM-False_e-0.1_B-500_task-button_topdown_absolute_pos-False_rep-alpha07v2-{}.csv'
test_HM_e_0_01_B_500_button_topdown_rel_alpha07v2 = 'HM-True_e-0.01_B-500_task-button_topdown_absolute_pos-False_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_50000_button_topdown_rel_alpha07v2 = 'HM-False_e-0.1_B-50000_task-button_topdown_absolute_pos-False_rep-alpha07v2-{}.csv'

tests = [test_NOHM_e_0_1_B_500_button_topdown_abs_alpha07v2, test_HM_e_0_01_B_50000_button_topdown_abs_alpha07v2, test_NOHM_e_0_1_B_500_button_topdown_rel_alpha07v2,test_HM_e_0_01_B_50000_button_topdown_rel_alpha07v2, test_NOHM_e_0_1_B_50000_button_topdown_rel_alpha07v2, test_HM_e_0_01_B_500_button_topdown_rel_alpha07v2]

test_HM_e_0_01_B_50000_hockey_abs_alpha07v2 = 'HM-True_e-0.01_B-50000_task-hockey_absolute_pos-True_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_500_hockey_abs_alpha07v2 = 'HM-False_e-0.1_B-500_task-hockey_absolute_pos-True_rep-alpha07v2-{}.csv'
test_HM_e_0_01_B_50000_hockey_rel_alpha07v2 = 'HM-True_e-0.01_B-50000_task-hockey_absolute_pos-False_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_500_hockey_rel_alpha07v2 = 'HM-False_e-0.1_B-500_task-hockey_absolute_pos-False_rep-alpha07v2-{}.csv'
test_HM_e_0_01_B_500_hockey_rel_alpha07v2 = 'HM-True_e-0.01_B-500_task-hockey_absolute_pos-False_rep-alpha07v2-{}.csv'
test_NOHM_e_0_1_B_50000_hockey_rel_alpha07v2 = 'HM-False_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha07v2-{}.csv'

tests = [test_NOHM_e_0_1_B_500_hockey_abs_alpha07v2, test_HM_e_0_01_B_50000_hockey_abs_alpha07v2, test_NOHM_e_0_1_B_500_hockey_rel_alpha07v2,test_HM_e_0_01_B_50000_hockey_rel_alpha07v2, test_NOHM_e_0_1_B_50000_hockey_rel_alpha07v2, test_HM_e_0_01_B_500_hockey_rel_alpha07v2]

tests = [test_NOHM_e_0_1_B_500_hockey_abs_alpha07v2, test_HM_e_0_01_B_50000_hockey_abs_alpha07v2, test_NOHM_e_0_1_B_500_button_topdown_abs_alpha07v2, test_HM_e_0_01_B_50000_button_topdown_abs_alpha07v2]

# 12-10-2021
test_HM_e_0_01_B_50000_button_rel_alpha07adjust = 'HM-True_e-0.01_B-50000_task-button_topdown_absolute_pos-False_rep-alpha07adjust-{}.csv'
test_HM_e_0_1_B_50000_button_rel_alpha07adjust = 'HM-True_e-0.1_B-50000_task-button_topdown_absolute_pos-False_rep-alpha07adjust-{}.csv'
test_HM_e_1_B_50000_button_rel_alpha07adjust = 'HM-True_e-1.0_B-50000_task-button_topdown_absolute_pos-False_rep-alpha07adjust-{}.csv'

tests = [test_HM_e_0_01_B_50000_button_rel_alpha07adjust, test_HM_e_0_1_B_50000_button_rel_alpha07adjust]

tests = [test_HM_e_0_01_B_50000_hockey_abs_alpha07v2, test_HM_e_0_1_B_50000_hockey_abs_shorter ]



test_HM_e_01_B_50000_hockey_abs_alpha09 = 'HM-True_e-0.01_B-50000_task-hockey_absolute_pos-True_rep-alpha09-{}.csv'
tests = [test_HM_e_0_01_B_50000_hockey_abs_alpha07v2, test_HM_e_0_1_B_50000_hockey_abs_shorter, test_HM_e_01_B_50000_hockey_abs_alpha09]

test_HM_e_01_B_50000_button_topdown_rel_alpha09 = 'HM-True_e-0.01_B-50000_task-button_topdown_absolute_pos-True_rep-alpha09-{}.csv'
tests = [test_HM_e_0_01_B_50000_button_topdown_abs_alpha07v2, test_HM_e_0_1_B_50000_button_topdown_abs_shorter, test_HM_e_01_B_50000_button_topdown_rel_alpha09]

test_HM_e_0_01_B_50000_hockey_rel_alpha09final = 'HM-True_e-0.01_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_01_B_500_hockey_rel_alpha09final = 'HM-True_e-0.01_B-500_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_50000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_500_hockey_rel_alpha09final = 'HM-True_e-0.1_B-500_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_01_B_50000_hockey_rel_alpha09final = 'HM-False_e-0.01_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_01_B_500_hockey_rel_alpha09final = 'HM-False_e-0.01_B-500_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_50000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_500_hockey_rel_alpha09final = 'HM-False_e-0.1_B-500_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_7500_hockey_rel_alpha09final = 'HM-False_e-0.1_B-7500_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-1.0_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_3000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
tests = [test_NOHM_e_0_01_B_50000_hockey_rel_alpha09final, test_NOHM_e_0_01_B_500_hockey_rel_alpha09final, test_NOHM_e_0_1_B_50000_hockey_rel_alpha09final, test_NOHM_e_0_1_B_500_hockey_rel_alpha09final, test_HM_e_0_01_B_50000_hockey_rel_alpha09final, test_HM_e_0_01_B_500_hockey_rel_alpha09final, test_HM_e_0_1_B_50000_hockey_rel_alpha09final, test_HM_e_0_1_B_500_hockey_rel_alpha09final, test_NOHM_e_0_1_B_7500_hockey_rel_alpha09final, test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final]
tests = [ test_NOHM_e_0_1_B_500_hockey_rel_alpha09final, test_NOHM_e_0_1_B_7500_hockey_rel_alpha09final, test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final,test_NOHM_e_0_1_B_50000_hockey_rel_alpha09final, test_NOHM_e_1_B_15000_hockey_rel_alpha09final, test_NOHM_e_0_1_B_3000_hockey_rel_alpha09final]

# # Hockey same e
# test_NOHM_e_0_1_B_3000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_50000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_3000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-3000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_15000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_50000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
#
# tests = [test_NOHM_e_0_1_B_3000_hockey_rel_alpha09final, test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final, test_NOHM_e_0_1_B_50000_hockey_rel_alpha09final, test_HM_e_0_1_B_3000_hockey_rel_alpha09final, test_HM_e_0_1_B_15000_hockey_rel_alpha09final, test_HM_e_0_1_B_50000_hockey_rel_alpha09final]

# Hockey same buffer
test_NOHM_e_0_01_B_15000_hockey_rel_alpha09final = 'HM-False_e-0.01_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-1.0_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_01_B_15000_hockey_rel_alpha09final = 'HM-True_e-0.01_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_15000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_1_B_15000_hockey_rel_alpha09final = 'HM-True_e-1.0_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'

tests = [test_NOHM_e_0_01_B_15000_hockey_rel_alpha09final, test_NOHM_e_0_1_B_15000_hockey_rel_alpha09final, test_NOHM_e_1_B_15000_hockey_rel_alpha09final, test_HM_e_0_01_B_15000_hockey_rel_alpha09final, test_HM_e_0_1_B_15000_hockey_rel_alpha09final, test_HM_e_1_B_15000_hockey_rel_alpha09final]



# # Hockey abs pos best of both methods:
# test_NOHM_e_1_B_15000_hockey_rel_alpha09final = 'HM-False_e-1.0_B-15000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_1_B_15000_hockey_abs_alpha09final = 'HM-False_e-1.0_B-15000_task-hockey_absolute_pos-True_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_30000_hockey_rel_alpha09final = 'HM-True_e-0.1_B-50000_task-hockey_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_30000_hockey_abs_alpha09final = 'HM-True_e-0.1_B-30000_task-hockey_absolute_pos-True_rep-alpha09final-{}.csv'
# tests = [test_NOHM_e_1_B_15000_hockey_rel_alpha09final, test_NOHM_e_1_B_15000_hockey_abs_alpha09final, test_HM_e_0_1_B_30000_hockey_rel_alpha09final,  test_HM_e_0_1_B_30000_hockey_abs_alpha09final ]

# # #
# # Drawer same e
# test_NOHM_e_0_1_B_3000_drawer_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_15000_drawer_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_30000_drawer_rel_alpha09final = 'HM-False_e-0.1_B-30000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_3000_drawer_rel_alpha09final = 'HM-True_e-0.1_B-3000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_15000_drawer_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_30000_drawer_rel_alpha09final = 'HM-True_e-0.1_B-30000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
#
# tests = [test_NOHM_e_0_1_B_3000_drawer_rel_alpha09final, test_NOHM_e_0_1_B_15000_drawer_rel_alpha09final, test_NOHM_e_0_1_B_30000_drawer_rel_alpha09final, test_HM_e_0_1_B_3000_drawer_rel_alpha09final, test_HM_e_0_1_B_15000_drawer_rel_alpha09final, test_HM_e_0_1_B_30000_drawer_rel_alpha09final]

# #
# #button_topdown same e
# test_NOHM_e_0_1_B_3000_button_topdown_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_15000_button_topdown_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_30000_button_topdown_rel_alpha09final = 'HM-False_e-0.1_B-30000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_3000_button_topdown_rel_alpha09final = 'HM-True_e-0.1_B-3000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_15000_button_topdown_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_30000_button_topdown_rel_alpha09final = 'HM-True_e-0.1_B-30000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
#
# tests = [test_NOHM_e_0_1_B_3000_button_topdown_rel_alpha09final, test_NOHM_e_0_1_B_15000_button_topdown_rel_alpha09final, test_NOHM_e_0_1_B_30000_button_topdown_rel_alpha09final, test_HM_e_0_1_B_3000_button_topdown_rel_alpha09final,test_HM_e_0_1_B_15000_button_topdown_rel_alpha09final, test_HM_e_0_1_B_30000_button_topdown_rel_alpha09final]
#

# # button_topdown same buffer
# test_NOHM_e_0_01_B_15000_button_topdown_rel_alpha09final = 'HM-False_e-0.01_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_15000_button_topdown_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_1_B_15000_button_topdown_rel_alpha09final = 'HM-False_e-1.0_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_01_B_15000_button_topdown_rel_alpha09final = 'HM-True_e-0.01_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_15000_button_topdown_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_HM_e_1_B_15000_button_topdown_rel_alpha09final = 'HM-True_e-1.0_B-15000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
#
# tests = [test_NOHM_e_0_01_B_15000_button_topdown_rel_alpha09final, test_NOHM_e_0_1_B_15000_button_topdown_rel_alpha09final, test_NOHM_e_1_B_15000_button_topdown_rel_alpha09final, test_HM_e_0_01_B_15000_button_topdown_rel_alpha09final, test_HM_e_0_1_B_15000_button_topdown_rel_alpha09final, test_HM_e_1_B_15000_button_topdown_rel_alpha09final]
#
# # Button abs pos
# test_NOHM_e_0_1_B_3000_button_topdown_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-button_topdown_absolute_pos-False_rep-alpha09final-{}.csv'
# test_NOHM_e_0_1_B_3000_button_topdown_abs_alpha09final = 'HM-False_e-0.1_B-3000_task-button_topdown_absolute_pos-True_rep-alpha09final-{}.csv'
# test_HM_e_0_1_B_30000_button_topdown_abs_alpha09final = 'HM-True_e-0.1_B-30000_task-button_topdown_absolute_pos-True_rep-alpha09final-{}.csv'
# tests = [test_NOHM_e_0_1_B_3000_button_topdown_rel_alpha09final, test_NOHM_e_0_1_B_3000_button_topdown_abs_alpha09final, test_HM_e_0_1_B_30000_button_topdown_rel_alpha09final, test_HM_e_0_1_B_30000_button_topdown_abs_alpha09final]

# Drawer same buffer
test_NOHM_e_0_01_B_15000_drawer_rel_alpha09final = 'HM-False_e-0.01_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_15000_drawer_rel_alpha09final = 'HM-False_e-0.1_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_1_B_15000_drawer_rel_alpha09final = 'HM-False_e-1.0_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_01_B_15000_drawer_rel_alpha09final = 'HM-True_e-0.01_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_15000_drawer_rel_alpha09final = 'HM-True_e-0.1_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_1_B_15000_drawer_rel_alpha09final = 'HM-True_e-1.0_B-15000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'

tests = [test_NOHM_e_0_01_B_15000_drawer_rel_alpha09final, test_NOHM_e_0_1_B_15000_drawer_rel_alpha09final, test_NOHM_e_1_B_15000_drawer_rel_alpha09final, test_HM_e_0_01_B_15000_drawer_rel_alpha09final, test_HM_e_0_1_B_15000_drawer_rel_alpha09final, test_HM_e_1_B_15000_drawer_rel_alpha09final]


## Drawer best conditions
test_NOHM_e_0_1_B_3000_drawer_rel_alpha09final = 'HM-False_e-0.1_B-3000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_NOHM_e_0_1_B_3000_drawer_abs_alpha09final = 'HM-False_e-0.1_B-3000_task-drawer_absolute_pos-True_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_30000_drawer_rel_alpha09final = 'HM-True_e-0.1_B-30000_task-drawer_absolute_pos-False_rep-alpha09final-{}.csv'
test_HM_e_0_1_B_30000_drawer_abs_alpha09final = 'HM-True_e-0.1_B-30000_task-drawer_absolute_pos-True_rep-alpha09final-{}.csv'

tests = [test_NOHM_e_0_1_B_3000_drawer_rel_alpha09final, test_NOHM_e_0_1_B_3000_drawer_abs_alpha09final, test_HM_e_0_1_B_30000_drawer_rel_alpha09final, test_HM_e_0_1_B_30000_drawer_abs_alpha09final]




cm = 1/2.54

fig, axs= plt.subplots(2, figsize=(17*cm, 17*cm))

#ax0 = axs
ax0 = axs[0]
ax2 = ax0.twiny()

ax1 = axs[1]
ax3 = ax1.twiny()
ax4 = ax1.twinx()

mujoco_timestep = 0.0125

def moving_average(x, w):
    # Define mask and store as an array
    mask = np.ones((1, w)) / w
    mask = mask[0, :]

    convolved_data = np.convolve(x, mask, 'same')

    y_padded = np.pad(x, (w // 2, w - 1 - w // 2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((w,)) / w, mode='valid')

    np.convolve(x, np.ones(w) / w, 'valid')
    return y_smooth

    #
    # # create a normalized Hanning window
    # windowSize = 500
    # window = np.hanning(windowSize)
    # window = window / window.sum()
    #
    #
    # return np.convolve(window, x, mode='valid')

    #
    # # b, a = signal.butter(3, 0.05, 'lowpass')
    # # filteredLowPass = signal.filtfilt(b, a, x)
    # # return filteredLowPass
    #
    # b, a = signal.butter(8, 0.05)
    # filtered = signal.filtfilt(b, a, x, method="gust")
    # return filtered



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
    if "soccer" in test:
        task = "soccer"
    if "sweep" in test:
        task = "sweep"
    if "drawer" in test:
        task = "drawer"

    if "org" in test:
        org = "yes"
    else:
        org = "no"





    timesteps_processed_list, success_processed_list, feedback_processed_list, pct_feedback_processed_list, \
    tau, e, human_model, buffer_size, abs_pos = postProcess(test, path)

    if human_model == 'yes':
        method = 'RCIdL'
    if human_model == 'no':
        method = 'D-COACH'



    a = np.array(success_processed_list)
    success_mean = np.mean(a, axis =0)
    success_std = np.std(a, axis=0)

    #print('success_mean', success_mean)

    a = np.array(feedback_processed_list)
    feedback_mean = np.mean(a, axis=0)

    a = np.array(pct_feedback_processed_list)
    pct_feedback_mean = np.mean(a, axis=0)


    a = np.array(timesteps_processed_list)
    timesteps_list = np.mean(a, axis=0)
    simulated_time = timesteps_list * mujoco_timestep /60


    # # Polyfit
    # z = np.polyfit(simulated_time, success_mean, 20)
    # p = np.poly1d(z)
    # fit_success = p(simulated_time)


    # Moving average
    fit_success2 = moving_average(success_mean, 1000)
    fit_success2_len = len(fit_success2)
    step = simulated_time[-1]/fit_success2_len
    simulated_time2 = np.arange(simulated_time[0],simulated_time[-1],step)


    # Remove points
    first_part_success = success_mean[0:10:1]
    first_part_success = list(first_part_success)


    second_part_success = success_mean[10::50]
    second_part_success = list(second_part_success)

    first_part_success.extend(second_part_success)
    success_mean_complete = first_part_success
    #success_mean_complete = first_part_success + second_part_success


    first_part_timesteps = timesteps_list[0:10:1]
    first_part_timesteps = list(first_part_timesteps)

    second_part_timesteps = timesteps_list[10::50]
    second_part_timesteps = list(second_part_timesteps)

    first_part_timesteps.extend(second_part_timesteps)
    timesteps_complete = first_part_timesteps


    # Moving average
    fit_success2 = moving_average(success_mean_complete, 25)


    fit_success2_len = len(fit_success2)
    step = timesteps_complete[-1]/fit_success2_len
    timesteps_every502 = np.arange(timesteps_complete[0],timesteps_complete[-1],step)










    e = '{:,g}'.format(e)
    buffer_size = '{:,g}'.format(buffer_size)

    #ax0.plot(simulated_time, success_mean, linewidth=1, zorder=0)
    #ax0.plot(timesteps_every50, success_every50, linewidth=1, zorder=0)
    ax0.plot(timesteps_every502, fit_success2, linewidth=2, label= method + ', Buffer: ' + str(buffer_size) + ', e: ' + str(e) + ', Absolute pos: ' + abs_pos)
    #ax0.plot(timesteps_complete, success_mean_complete, linewidth=1, zorder=0)


    ax0.fill_between(timesteps_list, success_mean - success_std, success_mean + success_std, alpha=0.05)

    #ax0.plot(simulated_time2, fit_success2, linewidth=2.0,
             #zorder=1, label= method + ', Buffer: ' + str(buffer_size) + ', e: ' + str(e) + ', Absolute pos: ' + abs_pos)
    #ax2.plot(timesteps_list, success_mean, linewidth=0.5)# alpha=0)
    # ax0.plot(simulated_time, success_mean, linewidth=0.5, zorder=0)
    #ax0.plot(simulated_time, fit_success2, linewidth=2, zorder=0)

    # axs[0].plot(simulated_time2, fit_success2, linewidth=1.5, zorder=1,
    #             label='H: ' + human_model  + ', Buffer size: ' + str(
    #                 buffer_size) + ', e: ' + str(e) +  ', Original obs: ' + org)

    ax0.set_ylabel('% of success')
    ax0.set_xlabel('min')

    title = "Evaluation of task: " + task
    ax0.set_title(title)
    ax0.legend(loc='lower right')
    ax0.set_ylim([0, 1.1])
    ax0.set_xlim([-1000, 74300])
    #ax0.set_xlim([-1, 15.5])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.set_xlabel('time steps')
    ax2.spines['bottom'].set_position(('outward', 40))
    plt.rc('legend', fontsize=8)  # legend fontsize

    #ax2.plot(timesteps_list, p(simulated_time), alpha=0)
    plt.xticks(rotation=5)
    ax2.set_ylim([0, 1.1])

    # Lower plot:
    a_list = list(range(0, 75001))
    P_h1 = []
    alpha_a1 = 0.7
    tau_t1 = 0.00001

    for i in a_list:
        P_h1.append(alpha_a1 * math.exp(-1 * tau_t1 * i))

    P_h2 = []
    alpha_a2 = 0.9
    tau_t2 = 0.00001

    for i in a_list:
        P_h2.append(alpha_a2 * math.exp(-1 * tau_t2 * i))

    P_h3 = []
    alpha_a3 = 0.9
    tau_t3 = 0.000015

    for i in a_list:


        P_h3.append(alpha_a3 * math.exp(-1 * tau_t3 * i))

    ax1.plot(simulated_time, pct_feedback_mean, linewidth=2.0, zorder=0,
             label='H: ' + human_model + ', Buffer size: ' + str(buffer_size) + ', e: ' + str(e) + 'task: ' + task)
    ax3.plot(timesteps_list, pct_feedback_mean, alpha=0)
    ax3.plot(a_list, P_h1, color='black', label='alpha: ' + str(alpha_a1) + ' tau: ' + str(tau_t1))
    ax3.plot(a_list, P_h2, color='red', label='alpha: ' + str(alpha_a2) + ' tau: ' + str(tau_t2))
    ax3.plot(a_list, P_h3, color='blue', label='alpha: ' + str(alpha_a3) + ' tau: ' + str(tau_t3))
    ax4.plot(simulated_time, feedback_mean)
    ax1.grid(linestyle='--')
    ax1.set_ylabel('% of feedback per episode')
    ax4.set_ylabel('Amount of feedback')
    ax1.set_xlabel('min')
    title = "Training feedback for task: " + task
    ax1.set_title(title)

    ax3.xaxis.set_ticks_position('bottom')
    ax3.xaxis.set_label_position('bottom')
    ax3.spines['bottom'].set_position(('outward', 40))
    ax3.set_xlabel('time steps')

    ax1.legend(loc='lower right')

    ax3.set_ylim([0, 1])
ax3.legend(loc='lower left')
plt.xticks(rotation=5)

ax0.grid(linestyle='--')
ax0.set_axisbelow(True)

# fig.subplots_adjust(top=1.5) # Space between the subplots

plt.show()