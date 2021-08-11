import rospy
import std_msgs.msg as std_msg
import time

def kukaee_call(data):
    global position
    position = data.data





def main():
    global position
    rospy.init_node('learner', anonymous=True)
    rospy.Subscriber("kuka_ee", std_msg.Float64MultiArray, kukaee_call)
    time.sleep(5)
    while True:
        print("position: ", position)

        if rospy.is_shutdown():
            print('shutdown')
            break
if __name__ == "__main__":


    main()