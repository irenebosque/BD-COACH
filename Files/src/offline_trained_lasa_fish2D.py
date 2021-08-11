"""
Loading and running a policy trained offline
"""

from collections import deque
import logging
import os
import random
from typing import List

import numpy as np
import tensorflow as tf
import gym
import math
import pickle

from dqn import DeepQNetwork
from policy_ensemble import PolicyEnsemble
from config import Config
from feedback import Feedback # CC
import time # CC
import rospy # CC
from std_msgs.msg import String # CC
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JoyFeedbackArray


f_RL = False # CC
input_size = (6,)
output_size = 7
alpha_1 = np.pi

def kukaee_call(data):
    global position
    position = data.data
    
def body1_call(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
    global xB1, yB1, zB1
    xB1 = data.pose.position.x
    yB1 = data.pose.position.y
    zB1 = data.pose.position.z

def reset() -> np.ndarray:
    global count, posrequest, restart_flag, nomove
    count = 0
    pose1.data = [0, np.pi / 2, 0, 0.65, 0.2, 0.6]
    posrequest = [0.65, 0.2, 0.6]
    pub_pos.publish(pose1)
    rospy.sleep(5.0)
    state = get_state()
    restart_flag = True
    nomove = False
    return [state]

def get_state() -> np.ndarray:
    global alpha, alpha_1, xB1, zB1, xB2_r_1, position,position_1, fish_1
    fish = [position[1] - xB1, position[0] - zB1]
    fish_vel = [fish[0] - fish_1[0], fish[1] - fish_1[1]]
    velocity = np.array(position) - np.array(position_1)
    state = np.array([ position[1], velocity[1], fish[0], fish[1], fish_vel[0], fish_vel[1]])
    position_1 = position
    fish_1 = fish

    return state

def step(action: np.ndarray) -> np.ndarray:
    global  count, posrequest, yB1, restart_flag, nomove, posnomove
    poslimit = [0.35, 0.9, -0.45, 0.45, 0.6, 0.6] # xlow xup ylow yup zlow zup
    delta = 0.7
    if not nomove:
        posnomove = [position[0], position[1]]
    if action == 0:
        posrequest[1] = position[1] - delta
        posrequest[0] = posnomove[0]
        nomove = False
    if action == 1:  # CC
        posrequest[1] = position[1] - delta/3
        posrequest[0] = posnomove[0]
        nomove = False
    if action == 2:
        posrequest[0] = posnomove[0]
        posrequest[1] = posnomove[1]
        nomove = True
    if action == 3:
        posrequest[1] = position[1] + delta/3
        posrequest[0] = posnomove[0]
        nomove = False
    if action == 4:  # CC
        posrequest[1] = position[1] + delta
        posrequest[0] = posnomove[0]
        nomove = False
    if action == 5:
        posrequest[0] = position[0] - delta*1.5
        posrequest[1] = posnomove[1]
        #print('entra-')
        nomove = False
    if action == 6:
        posrequest[0] = position[0] + delta*1.5
        posrequest[1] = posnomove[1]
        #print('entra+')
        nomove = False
    posrequest[0] = poslimit[0] if posrequest[0] < poslimit[0] else posrequest[0]
    posrequest[0] = poslimit[1] if posrequest[0] > poslimit[1] else posrequest[0]
    posrequest[1] = poslimit[2] if posrequest[1] < poslimit[2] else posrequest[1]
    posrequest[1] = poslimit[3] if posrequest[1] > poslimit[3] else posrequest[1]
    posrequest[2] = poslimit[4] if posrequest[2] < poslimit[4] else posrequest[2]
    posrequest[2] = poslimit[5] if posrequest[2] > poslimit[5] else posrequest[2]
    pose1.data = [0, np.pi / 2, 0, posrequest[0], posrequest[1], posrequest[2]]
    state = get_state()

    while restart_flag or yB1 > position[2]:
        pose1.data = [0, np.pi / 2, 0, 0.6, 0.2, 0.6]
        pub_pos.publish(pose1)
        rospy.sleep(0.1)
        #print(11111)
        state = get_state() #
        if yB1 > position[2]:
            restart_flag = False
    pub_pos.publish(pose1)
    reward = abs(state[3])
    count = count + 1
    done = True if count > 2000 else False

    return [state, reward, done]


def main():
    global alpha, alpha_1, position, position_1, pub_pos, pose1, posrequest, xB1, zB1, fish_1, h_joy
    alpha_1 = np.pi
    rospy.init_node('learner', anonymous=True)
    rospy.Subscriber("kuka_ee", Float64MultiArray, kukaee_call)
    rospy.Subscriber("/vrpn_client_node/RigidBody3/pose", PoseStamped, body3_call)
    rospy.Subscriber("/vrpn_client_node/RigidBody1/pose", PoseStamped, body1_call)
    rospy.Subscriber("/joy", Joy, joy_call)
    pub_pos = rospy.Publisher("iiwa/CustomControllers/command", Float64MultiArray, queue_size=1)
    pose1 = Float64MultiArray()
    posrequest = []
    rate = rospy.Rate(50)  # 10hz
    rospy.sleep(3)
    position_1 = position
    fish_1 = [position[1] - xB1, position[0] - zB1]
    state = reset()







    global_step = 1
    for episode in range(100):

        done = False
        step_count = 0

        e_reward = 0
        model_loss = 0
        #avg_reward = np.mean(last_n_game_reward)

        state = reset()
        print('Episode started')
        while not done:
            # actions, u_e = mainH.predict(state)
            #print(actions)
            #action = np.argmax(actions)


            # Get new state and reward from environment
            next_state, reward, done = step(action)


            state = next_state
            e_reward += reward
            step_count += 1



            global_step += 1
            rate.sleep()
            if rospy.is_shutdown():
                print('shutdown')
                break
        print('Episode finished')
        state = reset()

        #logger.info("Episode: {episode}  reward: {e_reward}  loss: {model_loss}  consecutive_{consecutive_len}_avg_reward: {avg_reward}")
        logger.info(
            "Episode: " + str(episode) + " reward: " + str(e_reward) + " loss: " + str(
                model_loss) + " consecutive_" + str(consecutive_len) + " avg_reward: " + str(avg_reward))
        rospy.loginfo(
            "Episode: " + str(episode) + " reward: " + str(e_reward) + " loss: " + str(
                model_loss) + " consecutive_" + str(consecutive_len) + " avg_reward: " + str(avg_reward))

        # CartPole-v0 Game Clear Checking Logic
        last_n_game_reward.append(e_reward)


        if rospy.is_shutdown():
            print('shutdown')
            break


if __name__ == "__main__":
    if FLAGS.model_name.startswith("MLP") and FLAGS.frame_size > 1:
        raise ValueError('do not support frame_size > 1 if model_name is MLP')

    main()