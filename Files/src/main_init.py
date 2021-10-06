import gym
from feedback import Feedback



#Position Controller:
#from kuka_env import KUKAenv
# Cartesian Controller:
from kukaenv import KUKAenv

import os
import numpy as np
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
from metaworld.policies.sawyer_plate_slide_v2_policy import SawyerPlateSlideV2Policy
from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_assembly_v2_policy import SawyerAssemblyV2Policy

from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy
from metaworld.policies.sawyer_shelf_place_v2_policy import SawyerShelfPlaceV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_soccer_v2_policy import SawyerSoccerV2Policy



from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy


"""
Script that initializes the variables used in the file main.py
"""


# Read program args
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='metaworld_low-dim_DCOACH', help='select file in config_files folder')
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
max_time_steps_episode = config_general.getint('max_time_steps_episode')
max_time_steps_per_repetition = float(config_general['max_time_steps_per_repetition'])
cut_feedback = float(config_general['cut_feedback'])
number_of_repetitions = config_general.getint('number_of_repetitions')
render_delay = float(config_general['render_delay'])
tau = float(config_general['tau'])
alpha = float(config_general['alpha'])
theta = float(config_general['theta'])
action_factor = float(config_general['action_factor'])
task = config_general['task']
human_teacher = config_general.getboolean('human_teacher')
absolute_positions = config_general.getboolean('absolute_positions')
oracle_teacher = config_general.getboolean('oracle_teacher')
evaluations_per_training = config_general.getint('evaluations_per_training')

dim_a=config_agent.getint('dim_a')
metaworld_env = config_general.getboolean('metaworld_env')
mountaincar_env = config_general.getboolean('mountaincar_env')
cartpole_env = config_general.getboolean('cartpole_env')
kuka_env = config_general.getboolean('kuka_env')
pendulum_env = config_general.getboolean('pendulum_env')


# Create Neural Network
neural_network = NeuralNetwork(transition_model_learning_rate=float(config_transition_model['learning_rate']),
                               lstm_hidden_state_size=config_transition_model.getint('lstm_hidden_state_size'),
                               load_transition_model=config_transition_model.getboolean('load_transition_model'),
                               load_policy=config_agent.getboolean('load_policy'),
                               dim_a=config_agent.getint('dim_a'),
                               dim_a_used=config_agent.getint('dim_a_used'),
                               dim_o=config_agent.getint('dim_o'),
                               network_loc=config_general['graph_folder_path'],
                               image_size=config_transition_model.getint('image_side_length'),
                               act_func_agent = config_general['act_func_agent'],
                               n_neurons_agent=config_transition_model.getint('n_neurons_agent'))


# Create Agent
agent = agent_selector(agent_type, config_agent)


# Joystick
# feedback_joystick_ROS = feedbackJoystickROS()



if kuka_env:
    env = KUKAenv()
    #env.init_varaibles()
    task_short = "kuka-park-cardboard"


if pendulum_env:

    env = gym.make(environment)  # create environment
    task_short = "pendulum"
    if render:
        env.render()
    feedback = Feedback(env=env,
                        key_type=config_feedback['key_type'],
                        h_up=config_feedback['h_up'],
                        h_down=config_feedback['h_down'],
                        h_right=config_feedback['h_right'],
                        h_left=config_feedback['h_left'],
                        h_null=config_feedback['h_null'])
    # # Instantiate model for the policy teacher
    # policy_oracle = neural_network.policy_model()
    # # Load the weights
    # weights = np.load('./weights/weights_oracle_policy_pendulum.npy', allow_pickle=True)
    # policy_oracle.set_weights(weights[-1])

if cartpole_env:

    env = gym.make(environment)  # create environment
    task_short = "cartpole"
    if render:
        env.render()
    feedback = Feedback(env=env,
                        key_type=config_feedback['key_type'],
                        h_up=config_feedback['h_up'],
                        h_down=config_feedback['h_down'],
                        h_right=config_feedback['h_right'],
                        h_left=config_feedback['h_left'],
                        h_null=config_feedback['h_null'])





if mountaincar_env:

    env = gym.make(environment)  # create environment

    # # Instantiate model for the policy teacher
    # policy_oracle = neural_network.policy_model()
    # # Load the weights
    # policy_oracle.load_weights('./weights/teacher_policy_mountaincar')

    task_short = "mountaincar"
    feedback = Feedback(env=env,
                        key_type=config_feedback['key_type'],
                        h_up=config_feedback['h_up'],
                        h_down=config_feedback['h_down'],
                        h_right=config_feedback['h_right'],
                        h_left=config_feedback['h_left'],
                        h_null=config_feedback['h_null'])

elif metaworld_env:


    # Create Environment
    task = task.strip('"')
    plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
    env = plate_slide_goal_observable_cls()

    # Create Oracle policy
    if task == "drawer-open-v2-goal-observable":
        policy_oracle = SawyerDrawerOpenV2Policy()
        task_short = "drawer"
    elif task == "button-press-v2-goal-observable":
        policy_oracle = SawyerButtonPressV2Policy()
        task_short = "button"
        task_with_gripper = False
    elif task == "reach-v2-goal-observable":
        policy_oracle = SawyerReachV2Policy()
        task_short = "reach"
        task_with_gripper = False
    elif task == "plate-slide-v2-goal-observable":
        policy_oracle = SawyerPlateSlideV2Policy()
        task_short = "hockey"
        task_with_gripper = False
    elif task == "button-press-topdown-v2-goal-observable":
        policy_oracle = SawyerButtonPressTopdownV2Policy()
        task_short = "button_topdown"
        task_with_gripper = False
    elif task == "push-v2-goal-observable":
        policy_oracle = SawyerPushV2Policy()
        task_short = "push"
    elif task == "peg_insertion_side-v2-goal-observable":
        policy_oracle = SawyerPegInsertionSideV2Policy()
        task_short = "peg"
    elif task == "door-open-v2-goal-observable":
        policy_oracle = SawyerDoorOpenV2Policy()
        task_short = "door"
        task_with_gripper = False
    elif task == "assembly-v2-goal-observable":
        policy_oracle = SawyerAssemblyV2Policy()
        task_short = "assembly"
        task_with_gripper = True
    elif task == "basketball-v2-goal-observable":
        policy_oracle = SawyerBasketballV2Policy()
        task_short = "basketball"
        task_with_gripper = True
    elif task == "shelf-place-v2-goal-observable":
        policy_oracle = SawyerShelfPlaceV2Policy()
        task_short = "shelf"
        task_with_gripper = True
    elif task == "soccer-v2-goal-observable":
        policy_oracle = SawyerSoccerV2Policy()
        task_short = "soccer"
        task_with_gripper = False
    elif task == "pick-place-v2-goal-observable":
        policy_oracle = SawyerPickPlaceV2Policy()
        task_short = "pick"
        task_with_gripper = True

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
