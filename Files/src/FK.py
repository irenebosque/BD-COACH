import time # CC
import numpy as np
import logging
import os
import rospy # CC
from tf import TransformListener
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math




def jointposition_call(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard position %s", data.position)
    global a1, a2, a3, a4
    a1 = data.position[0]
    a2 = data.position[1]
    a3 = data.position[2]
    a4 = data.position[3]

def get_ee_pose(tf):
    print(1.0)
    while not (tf.frameExists("iiwa_link_7")) :
        time.sleep(0.1)
        print(2.0)
        if tf.frameExists("iiwa_link_7") and tf.frameExists("iiwa_link_0"):
            t = tf.getLatestCommonTime("iiwa_link_7", "iiwa_link_0")
            position, quaternion = tf.lookupTransform("iiwa_link_0", "iiwa_link_7", t)
            pose1.data = [0, 1.5, 0, position[0], position[1], position[2]]
            pub_pos.publish(pose1)
            print(3.0)

    if tf.frameExists("iiwa_link_7") and tf.frameExists("iiwa_link_0"):
        t = tf.getLatestCommonTime("iiwa_link_7", "iiwa_link_0")
        position, quaternion = tf.lookupTransform("iiwa_link_0", "iiwa_link_7", t)
        return position, quaternion
    else:
        return None

def main():
    global pub_pos, pose1

    def euler_from_quaternion(x, y, z, w):
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
    rospy.init_node('FK', anonymous=True)
    rospy.Subscriber("iiwa/joint_states", JointState, jointposition_call)
    pub_pos = rospy.Publisher("iiwa/CustomControllers/command", Float64MultiArray, queue_size=1)
    pub_posee = rospy.Publisher("kuka_ee", Float64MultiArray, queue_size=1)
    pose1 = Float64MultiArray()
    ee = Float64MultiArray()
    rate = rospy.Rate(100)  # 100hz
    tf = TransformListener()
    print(0.0)
    time.sleep(3)
    while True:
        position, quaternion = get_ee_pose(tf)

        orientation = euler_from_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

        print("orientation: ", orientation)
        ee.data = position
        pub_posee.publish(ee)
        print(position)
        rate.sleep()
        if rospy.is_shutdown():
            print('shutdown')
            break
    print('finished')


if __name__ == "__main__":
    main()