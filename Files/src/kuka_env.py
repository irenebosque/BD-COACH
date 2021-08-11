
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
        global pose1, pub_pos, alpha_1, h, obj_pos, pendulum_pose

        rospy.Subscriber('iiwa/joint_states', JointState, self.joint_call)
        rospy.Subscriber("/vrpn_client_node/RigidBody3/pose", PoseStamped, self.body_call)
        rospy.Subscriber("/joy", Joy, self.joy_call)
        rospy.Subscriber("kuka_ee", std_msg.Float64MultiArray, self.kukaee_call)
        pub_pos = rospy.Publisher('iiwa/PositionController/command', std_msg.Float64MultiArray, queue_size=1)
        obj_pos = rospy.Publisher('pendulo', std_msg.Float64MultiArray, queue_size=1)

        pose1 = std_msg.Float64MultiArray()
        pendulum_pose = std_msg.Float64MultiArray()
        alpha_1 = np.pi
        time.sleep(3)



    def kukaee_call(self, data):
        global positionee
        positionee = data.data

    def joint_call(self, data):

        global position, velocity
        position = data.position
        velocity = data.velocity

    def joy_call(self, data):
        global h, button_pause, button_A
        h = data.axes[3]
        button_pause = data.buttons[5]
        button_A = data.buttons[0]
        button_B = data.buttons[1]



    def get_h(self):
        global h, button_A

        if button_A == 1:
            return [10]


        return [-1 * h]


    def body_call(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
        global alpha, xB, yB
        xB = data.pose.position.x
        yB = data.pose.position.y
        x = data.pose.orientation.x
        y = data.pose.orientation.y
        z = data.pose.orientation.z
        w = data.pose.orientation.w
        roll, pitch, yaw = self.euler_from_quaternion(x, y, z, w)
        pendulum_pose.data = [roll, pitch, yaw ]
        obj_pos.publish(pendulum_pose)
        #alpha = yaw + np.pi # This alpha is NOT an angle, is a distance
        #alpha = alpha - 2 * np.pi if alpha > np.pi else alpha

    def get_state(self) -> np.ndarray:
        global alpha, alpha_1, xB, yB, positionee
        """l = 0.5755
        alpha = np.arccos(np.clip(((yB - positionee[2]) / l), -1.0, 1.0))
        alpha = alpha if (xB > positionee[1]) else -1 * alpha
        alpha_1 = alpha_1 + 2 * np.pi if (alpha > np.pi / 2 and alpha_1 < -np.pi / 2) else alpha_1
        alpha_1 = alpha_1 - 2 * np.pi if (alpha < -np.pi / 2 and alpha_1 > np.pi / 2) else alpha_1"""
        # alphap = alpha - alpha_1

        # alphap = np.clip(10*(alpha - alpha_1), -1,1)#so far the best
        # I think alpha is the distance between both ends of the pendulum in the axis of the table
        alpha = xB - positionee[1]

        # l = 0.815
        # angle = np.arccos(np.clip(((yB - positionee[2]) / l), -1.0, 1.0))
        # alpha = np.sin(angle) * l


        alphap = alpha - alpha_1
        # velocity = np.array(position) - np.array(position_1)
        state = np.array([alpha, alphap, position[0], velocity[0]])
        alpha_1 = alpha
        return state

    def step(self, action: np.ndarray) -> np.ndarray:
        global count, posrequest, yB1, restart_flag, nomove, posnomove, button_pause



        poslimit = [-np.pi / 7, np.pi / 7]  # xlow xup ylow yup zlow zup
        #poslimit = [-np.pi / 12, np.pi / 12]  # xlow xup ylow yup zlow zup
        delta = 0.075


        posrequest0 = position[0] + delta * action
        posrequest2 = position[2] + delta * action

        if button_pause == 1:  # pause episode while holding it
            posnomove0 = position[0]
            posrequest0 = posnomove0
            posnomove2 = position[2]
            posrequest2 = posnomove2

        elif  button_pause == 0:
            count = count + 1

        # if button_B == 1:
        #     posnomove0 = position[0]
        #     posrequest0 = posnomove0
        #     posnomove2 = position[2]
        #     posrequest2 = posnomove2
        #
        # if button_A == 1:
        #     pause_Flag = False
        #


        posrequest0 = poslimit[0] if posrequest0 < poslimit[0] else posrequest0
        posrequest0 = poslimit[1] if posrequest0 > poslimit[1] else posrequest0
        posrequest2 = poslimit[0] if posrequest2 < poslimit[0] else posrequest2
        posrequest2 = poslimit[1] if posrequest2 > poslimit[1] else posrequest2
        print("posrequest2: ", posrequest2)
        #pose1.data = [posrequest0, 0.5, posrequest2, -np.pi / 2, 0, -0.5, 0]

        pose1.data = [posrequest0, np.pi / 4, 0, -np.pi / 4, -np.pi/2, 0, 0]




        state = self.get_state()


        while abs(state[0]) > np.pi / 3:
            # pose1.data = [0, np.pi / 2, 0, 0.6, 0.2, 0.6]
            # pub_pos.publish(pose1)
            rospy.sleep(0.05)
            # print(11111)
            state = get_state()  #
            if abs(state[0]) > np.pi / 12:
                state[0] = np.pi / 2
        pub_pos.publish(pose1)
        count = count + 1
        reward = abs(state[0])
        count = count + 1
        done = True if count > 2000 else False
        info = []

        return [state, reward, done, info]

    def reset(self) -> np.ndarray:
        global count, posrequest, restart_flag, nomove, position
        count = 0
        delta = abs(0 - position[0]) / 30
        for itera in range(100):
            pose1.data = [0, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, 0, 0]
            if 0 - position[0] > (delta):
                #pose1.data = [position[2] + delta, 0.5, position[2] + delta, -np.pi / 2, 0, -0.5, 0]
                pose1.data = [position[0] + delta, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, 0, 0]
            if 0 - position[0] < -delta:
                #pose1.data = [position[2] - delta, 0.5, position[2] - delta, -np.pi / 2, 0, -0.5, 0]
                pose1.data = [position[0] - delta, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, 0, 0]
            pub_pos.publish(pose1)
            rospy.sleep(0.05)
        posrequest = 0
        #pose1.data = [0, 0.5, 0, -np.pi / 2, 0, -0.50, 0]
        pose1.data = [0, np.pi / 4, 0, -np.pi / 4, -np.pi / 2, 0, 0]
        pub_pos.publish(pose1)
        rospy.sleep(1.0)
        state = self.get_state()
        self.get_h()
        restart_flag = True
        nomove = False
        return [state]

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians






