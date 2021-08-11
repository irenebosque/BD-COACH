
import std_msgs.msg as std_msg
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
import sensor_msgs.msg as sensor_msg
import rospy
import numpy as np
import tensorflow as tf
import math
import time


class KUKAenv:
    def __init__(self):
        global pose1, pub_pos, alpha_1, h, obj_pos, pendulum_pose, fish_1, position, position_1, xB1, zB1

        #rospy.Subscriber('iiwa/joint_states', JointState, self.joint_call)
        rospy.Subscriber("/joy", Joy, self.joy_call)
        rospy.Subscriber("kuka_ee", std_msg.Float64MultiArray, self.kukaee_call)
        rospy.Subscriber("/vrpn_client_node/RigidBody3/pose", PoseStamped, self.body_call)


        pub_pos = rospy.Publisher('iiwa/CustomControllers/command', std_msg.Float64MultiArray, queue_size=1)
        #obj_pos = rospy.Publisher('pendulo', std_msg.Float64MultiArray, queue_size=1)
        print('eooo')
        rospy.sleep(3)


        position = [-0.003454930158990385, -0.0097650557236241, 1.2034581470005148]
        fish_1 = [0, 0]
        #position_1 = position
        #fish_1 = [position[1] - xB1, position[0] - zB1]

        pose1 = std_msg.Float64MultiArray()
        pendulum_pose = std_msg.Float64MultiArray()
        alpha_1 = np.pi

    def init_varaibles(self):
        global position, position_1, fish_1, xB1, zB1
        position_1 = position
        fish_1 = [position[1] - xB1, position[0] - zB1]




    def kukaee_call(self, data):
        global position
        position = data.data
        #print("position", position)


    # def joint_call(self, data):
    #
    #     global position, velocity
    #     position = data.position
    #     velocity = data.velocity

    def joy_call(self, data):
        global h, button_pause
        h = data.axes[3]
        button_pause = data.buttons[5]
        print("h: ", h)



    def get_h(self):
        global h

        return [-1 * h]

    def body_call(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
        global xB1, yB1, zB1
        xB1 = data.pose.position.x
        yB1 = data.pose.position.y
        zB1 = data.pose.position.z





    def get_state(self) -> np.ndarray:
        global alpha, alpha_1, xB1, zB1, xB2_r_1, position, position_1, fish_1
        fish = [position[1] - xB1, position[0] - zB1]
        fish_vel = [fish[0] - fish_1[0], fish[1] - fish_1[1]]
        velocity = np.array(position) - np.array(position_1)
        #state = np.array([position[1], velocity[1], fish[0], fish[1], fish_vel[0], fish_vel[1]])
        state = np.array([position[1], velocity[1], fish[1], fish_vel[1]])
        position_1 = position
        fish_1 = fish

        return state

    def step(self, action: np.ndarray) -> np.ndarray:
        global count, posrequest, yB1, restart_flag, nomove, posnomove
        poslimit = [0.35, 0.9, -0.45, 0.45, 0.6, 0.6]  # xlow xup ylow yup zlow zup
        delta = 0.25 #0.7
        if not nomove:
            posnomove = [position[0], position[1]]
        if action == 0:
            posrequest[1] = position[1] - delta
            posrequest[0] = posnomove[0]
            nomove = False
        if action == 1:  # CC
            posrequest[1] = position[1] - delta / 3
            posrequest[0] = posnomove[0]
            nomove = False
        if action == 2:
            posrequest[0] = posnomove[0]
            posrequest[1] = posnomove[1]
            nomove = True
        if action == 3:
            posrequest[1] = position[1] + delta / 3
            posrequest[0] = posnomove[0]
            nomove = False
        if action == 4:  # CC
            posrequest[1] = position[1] + delta
            posrequest[0] = posnomove[0]
            nomove = False
        if action == 5:
            posrequest[0] = position[0] - delta * 1.5
            posrequest[1] = posnomove[1]
            # print('entra-')
            nomove = False
        if action == 6:
            posrequest[0] = position[0] + delta * 1.5
            posrequest[1] = posnomove[1]
            # print('entra+')
            nomove = False
        posrequest[0] = poslimit[0] if posrequest[0] < poslimit[0] else posrequest[0]
        posrequest[0] = poslimit[1] if posrequest[0] > poslimit[1] else posrequest[0]
        posrequest[1] = poslimit[2] if posrequest[1] < poslimit[2] else posrequest[1]
        posrequest[1] = poslimit[3] if posrequest[1] > poslimit[3] else posrequest[1]
        posrequest[2] = poslimit[4] if posrequest[2] < poslimit[4] else posrequest[2]
        posrequest[2] = poslimit[5] if posrequest[2] > poslimit[5] else posrequest[2]
        pose1.data = [0, np.pi / 2, 0, posrequest[0], posrequest[1], posrequest[2]]
        pose1.data = [0, np.pi / 2, 0, 0.65, posrequest[1], 0.6]
        state = self.get_state()

        # while restart_flag or yB1 > position[2]:
        #     pose1.data = [0, np.pi / 2, 0, 0.65, 0.2, 0.6]
        #     pub_pos.publish(pose1)
        #     rospy.sleep(0.1)
        #     # print(11111)
        #     state = self.get_state()  #
        #     if yB1 > position[2]:
        #         restart_flag = False
        pub_pos.publish(pose1)
        reward = abs(state[3])
        count = count + 1
        done = True if count > 2000 else False

        info = []

        return [state, reward, done, info]

    def reset(self) -> np.ndarray:
        global count, posrequest, restart_flag, nomove
        count = 0
        pose1.data = [0, np.pi / 2, 0, 0.65, 0.2, 0.6]
        posrequest = [0.65, 0.2, 0.6]
        pub_pos.publish(pose1)
        rospy.sleep(5.0)
        state = self.get_state()
        restart_flag = True
        nomove = False
        return [state]






