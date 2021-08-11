import std_msgs.msg as std_msg
import sensor_msgs.msg as sensor_msg
from tf import TransformListener
import rospy
import time
import numpy as np
import sys


class PoseVelocity:
    def __init__(self):
        self.time_m1 = None
        self.pos_m1 = None
        self.tf = TransformListener()
        self.pose_velocity = rospy.Publisher('iiwa/pose_velocity', std_msg.Float64MultiArray, queue_size=10)

    def get_ee_pose(self):
        if self.tf.frameExists("iiwa_link_7") and self.tf.frameExists("iiwa_link_0"):
            t = self.tf.getLatestCommonTime("iiwa_link_7", "iiwa_link_0")
            position, quaternion = self.tf.lookupTransform("iiwa_link_0", "iiwa_link_7", t)
            return position
        else:
            return None

    def get_ee_vel(self, position):
        current_time = rospy.get_time()

        if self.time_m1 == None:
            self.time_m1 = -1
            self.pos_m1 = position

        velocity = (position - self.pos_m1) / (current_time - self.time_m1)
        self.time_m1 = current_time
        self.pos_m1 = position
        return velocity

    def get_state(self):
        ee_position = np.array(self.get_ee_pose())
        ee_velocity = self.get_ee_vel(ee_position)
        return ee_position, ee_velocity

    def publish_topic(self, position, velocity):
        msg = std_msg.Float64MultiArray()
        msg.data = [position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]]
        self.pose_velocity.publish(msg)


if __name__ == '__main__':
    rospy.init_node('pose_velocity')
    pose_velocity = PoseVelocity()
    rate = rospy.Rate(1)
    rate.sleep()  # apparently we need to run this first for the reset to work

    while True:
        ee_position, ee_velocity = pose_velocity.get_state()
        pose_velocity.publish_topic(ee_position, ee_velocity)
        rospy.sleep(0.05)