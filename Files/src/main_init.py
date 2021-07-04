import gym
from feedback import Feedback
import os
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # switch to CPU
from buffer import Buffer
from agents.selector import agent_selector
#from transition_model import TransitionModel
from neural_network import NeuralNetwork
from tools.functions import load_config_data
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy



"""
Script that initializes the variables used in the file main.py
"""


# Read program args
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='metaworld-plate-slide-v2-goal-observable', help='select file in config_files folder')
parser.add_argument('--exp-num', default='-1')
args = parser.parse_args()

config_file = args.config_file
exp_num = args.exp_num

# Load from config files
config = load_config_data('config_files/' + config_file + '.ini')
config_general = config['GENERAL']
config_transition_model = config['TRANSITION_MODEL']
config_agent = config['AGENT']
config_feedback = config['FEEDBACK']

agent_type = config_agent['agent']
environment = config_general['environment']
transition_model_type = config_transition_model['transition_model']
# eval_save_path = 'results/' + environment + '_' + agent_type + '_' + transition_model_type + '/'
eval_save_path = '/home/irene/DashBoard/DCOACH-with-off-policy-human-model/Files/src/results/'

render = config_general.getboolean('render')
count_down = config_general.getboolean('count_down')
save_results = config_general.getboolean('save_results')
evaluation = config_general.getboolean('evaluate')
save_policy = config_agent.getboolean('save_policy')
save_transition_model = config_transition_model.getboolean('save_transition_model')
max_num_of_episodes = config_general.getint('max_num_of_episodes')
max_time_steps_episode = float(config_general['max_time_steps_episode'])
max_time_steps_per_repetition = float(config_general['max_time_steps_per_repetition'])
number_of_repetitions = config_general.getint('number_of_repetitions')
render_delay = float(config_general['render_delay'])
tau = float(config_general['tau'])
alpha = float(config_general['alpha'])
theta = float(config_general['theta'])
task = config_general['task']



# Create Neural Network
neural_network = NeuralNetwork(transition_model_learning_rate=float(config_transition_model['learning_rate']),
                               lstm_hidden_state_size=config_transition_model.getint('lstm_hidden_state_size'),
                               load_transition_model=config_transition_model.getboolean('load_transition_model'),
                               load_policy=config_agent.getboolean('load_policy'),
                               dim_a=config_agent.getint('dim_a'),
                               dim_a_used=config_agent.getint('dim_a_used'),
                               dim_o=config_agent.getint('dim_o'),
                               network_loc=config_general['graph_folder_path'],
                               image_size=config_transition_model.getint('image_side_length'))


# Create Environment
task = task.strip('"')
plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
env = plate_slide_goal_observable_cls()

# Create Agent
agent = agent_selector(agent_type, config_agent)

# Create Oracle policy
if task == "drawer-open-v2-goal-observable":
    policy_oracle = SawyerDrawerOpenV2Policy()
    task_short = "drawer"
elif task == "button-press-v2-goal-observable":
    policy_oracle = SawyerButtonPressV2Policy()
    task_short = "button"
elif task == "reach-v2-goal-observable":
    policy_oracle = SawyerReachV2Policy()
    task_short = "reach"
elif task == "plate-slide-v2-goal-observable":
    policy_oracle = SawyerReachV2Policy()
    task_short = "hockey"









# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
