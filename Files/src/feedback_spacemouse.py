import sensor_msgs.msg as sensor_msg
import rospy


class feedbackSpacemouse:
    def __init__(self):

        self.spacenav_state = None
        self.spacenav_buttons = None
        rospy.Subscriber('/spacenav/joy', sensor_msg.Joy, self._callback_recorder,
                         queue_size=10)

    def _callback_recorder(self, data):
        self.spacenav_state = data.axes

        self.spacenav_buttons = data.buttons

    def get_h(self):

        return self.spacenav_state

    def ask_for_done(self):
        if self.spacenav_buttons[0] and self.spacenav_buttons[1]:
            done = True
        else:
            done = False
        return done