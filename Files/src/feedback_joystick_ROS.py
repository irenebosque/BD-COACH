from sensor_msgs.msg import Joy
import rospy
import time


class feedbackJoystickROS:
    def __init__(self):
        global  h
        print("HOOOOOLI")
        rospy.Subscriber("/joy", Joy, self.joy_call)
        time.sleep(3)



    def joy_call(self, data):
        global h, button_pause
        h = data.axes[0]
        button_pause = data.buttons[5]



    def get_h(self):
        global h

        return [h]


