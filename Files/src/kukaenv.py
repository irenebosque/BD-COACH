
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
      
        #rospy.Subscriber('iiwa/joint_states', JointState, self.joint_call)
        rospy.Subscriber("/joy", Joy, self.joy_call)
        rospy.Subscriber("kuka_ee", std_msg.Float64MultiArray, self.kukaee_call)
        rospy.Subscriber("/vrpn_client_node/RigidBody7/pose", PoseStamped, self.body_call)


        self.pub_pos = rospy.Publisher('iiwa/CustomControllers/command', std_msg.Float64MultiArray, queue_size=1)
        #obj_pos = rospy.Publisher('pendulo', std_msg.Float64MultiArray, queue_size=1)
        print('eooo')
        rospy.sleep(3)

        self.position = [-0.003454930158990385, -0.0097650557236241, 1.2034581470005148]
        self.fish_1 = [0, 0]
        self.position_1 = self.position
        # self.fish_1 = [self.position[1] - self.xB1, self.position[0] - self.zB1]

        self.pose1 = std_msg.Float64MultiArray()
      
        self.alpha_1 = np.pi
        self.posnomove = 0






    def kukaee_call(self, data):
 
        self.position = data.data
        #print("self.position", self.position)


    # def joint_call(self, data):
    #
    #     global self.position, velocity
    #     self.position = data.self.position
    #     velocity = data.velocity

    def joy_call(self, data):
    
        self.h_xB1 = data.axes[3]
        self.h_zB1 = data.axes[4]
        self.button_pause = data.buttons[5]
        self.button_slow = data.buttons[0]
        self.button_finish_episode = data.buttons[1]




    # def get_h(self):
    #     if self.button_finish_episode == 1:
    #         return True
    #     elif self.button_slow == 1:
    #         return [10, 10]
    #     else:
    #
    #         return [self.h_zB1*-1, self.h_xB1*-1]

    def get_h(self):
        if self.button_finish_episode == 1:
            doneButton = True
        else:
            doneButton = False


        # if self.button_slow == 1:
        #     h = [10, 10]

        h = [self.h_zB1 * -1, self.h_xB1 * -1]


        return h, doneButton

    def body_call(self, data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
  
        self.xB1 = data.pose.position.x
        self.yB1 = data.pose.position.y
        self.zB1 = data.pose.position.z
        x = data.pose.orientation.x
        y = data.pose.orientation.y
        z = data.pose.orientation.z
        w = data.pose.orientation.w
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(x, y, z, w)
        #print('pitch: ', self.pitch)
        #
        # print('roll: ', self.roll)
        #
        # print('yaw: ', self.yaw)





    def get_state(self) -> np.ndarray:
        #para la bola del mundo no necesitas la profundidad
    
        self.fish = [self.position[1] - self.xB1, self.position[0] - self.zB1]
        self.fish_vel = [self.fish[0] - self.fish_1[0], self.fish[1] - self.fish_1[1]]
        self.velocity = np.array(self.position) - np.array(self.position_1)
        #self.state = np.array([self.position[1], self.velocity[1], self.fish[0], self.fish[1], self.fish_vel[0], self.fish_vel[1]])
        #self.state = np.array([self.position[1], self.velocity[1], self.fish[1], self.fish_vel[1]])

        # Scale state dimensions to a range -1 to 1
        pitch_scale = self.pitch/((np.pi/2)*0.9)

        max_velocity = 0.009*1.1






        # For the football with box:
        #self.state = np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1], self.xB1, self.zB1, self.pitch])
        self.state = np.array([self.fish[0], self.fish[1], self.velocity[0]/max_velocity, self.velocity[1]/max_velocity, self.xB1, self.zB1, pitch_scale])

        print("State", self.state)


        self.position_1 = self.position
        self.fish_1 = self.fish

        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
      
        #poslimit = [0.35, 0.9, -0.45, 0.45, 0.6, 0.6]  # xlow xup ylow yup zlow zup

        # For football with box:
        poslimit = [0.25, 0.80, -0.45, 0.45, 0.16, 0.16]  # xlow xup ylow yup zlow zup
        delta = 0.05 #0.7

        Delta_d = delta * action
        print("Delta_d before", Delta_d )
        Delta_d_min = 0.02

        for i in range(2):
            if Delta_d[i] > 0:
                Delta_d[i] = Delta_d[i] + Delta_d_min
            elif Delta_d[i] < 0:
                Delta_d[i] = Delta_d[i] - Delta_d_min

        print("Delta_d after", Delta_d)


        self.posrequest[1] = self.position[1] +  Delta_d[1]
        self.posrequest[0] = self.position[0] +  Delta_d[0]
        # self.posrequest[1] = self.position[1] + delta * action[1]
        # self.posrequest[0] = self.position[0] + delta * action[0]
        # self.posrequest[1] = self.position[1] + delta * 0.1
        # self.posrequest[0] = self.position[0] + delta * 0.1
        print("delta: ", delta*0.1)


        if self.button_pause == 1:  # pause episode while holding it
            posnomoveY = self.position[1]
            posnomoveX = self.position[0]
            self.posrequest[1] = posnomoveY
            self.posrequest[0] = posnomoveX

        elif  self.button_pause == 0:
            self.count = self.count + 1


        self.posrequest[1] = poslimit[2] if self.posrequest[1] < poslimit[2] else self.posrequest[1]
        self.posrequest[1] = poslimit[3] if self.posrequest[1] > poslimit[3] else self.posrequest[1]

        self.posrequest[0] = poslimit[0] if self.posrequest[0] < poslimit[0] else self.posrequest[0]
        self.posrequest[0] = poslimit[1] if self.posrequest[0] > poslimit[1] else self.posrequest[0]

        #self.pose1.data = [0, np.pi / 2, 0, self.posrequest[0], self.posrequest[1], self.posrequest[2]]
        #self.pose1.data = [0, np.pi / 2, 0, 0.65, self.posrequest[1], 0.6]
        # For football with box:
        self.pose1.data = [0, np.pi, 0, self.posrequest[0], self.posrequest[1], 0.14]
        #self.pose1.data = [0, np.pi, 0, 0.5, self.posrequest[1], 0.16]


        self.state = self.get_state()

        # while self.restart_flag or yB1 > self.position[2]:
        #     self.pose1.data = [0, np.pi / 2, 0, 0.65, 0.2, 0.6]
        #     pub_pos.publish(pose1)
        #     rospy.sleep(0.1)
        #     # print(11111)
        #    self.state = self.get_state()  #
        #     if yB1 > self.position[2]:
        #         self.restart_flag = False
        self.pub_pos.publish(self.pose1)
        reward = abs(self.state[3])
        self.count = self.count + 1
        info = {"success": 0}


        done = True if self.count > 2000 else False

        radio = 0.02



        target_point = [0.626, 0.009]
        print('coor: ', self.zB1, self.xB1)

        if (self.zB1 > target_point[0] - radio and self.zB1 < target_point[0] + radio) and (self.xB1 > target_point[1] - radio and self.xB1 < target_point[1] + radio):
            print('trurru')
            info = {"success":1}





        #info = []



        return [self.state, reward, done, info]

    def reset(self) -> np.ndarray:

        self.count = 0
        # # Pendulum
        # self.pose1.data = [0, np.pi, 0, 0.65, 0.2, 0.6]
        # self.posrequest = [0.65, 0.2, 0.6]

        # Football with box:
        #self.pose1.data = [-np.pi, 0, np.pi, 0.65, 0.06, 0.16]
        self.pose1.data = [0, np.pi, 0, 0.6, 0.4, 0.14]
        #self.pose1.data = [0, np.pi, 0, 0.35, 0.45, 0.16]

        self.posrequest = [ 0.6, 0.4, 0.14]
        #self.posrequest = [0.35, 0.45, 0.16]



        self.pub_pos.publish(self.pose1)
        rospy.sleep(10.0)
        self.state = self.get_state()
        self.restart_flag = True
        self.nomove = False
        return [self.state]
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







