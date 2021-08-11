"""
Combining HG-DAgger and Double DQN, the policy is an ensenmble of NNs
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


flags = tf.app.flags
flags.DEFINE_float('discount_rate', 0.99, 'Initial discount rate.')
flags.DEFINE_integer('replay_memory_length', 50000, 'Number of replay memory episode.')
flags.DEFINE_integer('target_update_count', 5, 'DQN Target Network update count.')
flags.DEFINE_integer('max_episode_count', 5000, 'Number of maximum episodes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. (Must divide evenly into the dataset sizes)')#64
flags.DEFINE_integer('frame_size', 1, 'Frame size. (Stack env\'s observation T-n ~ T)')
flags.DEFINE_string('model_name', 'MLPv1', 'DeepLearning Network Model name (MLPv1, ConvNetv1)')
flags.DEFINE_float('learning_rate', 0.01, 'Batch size. (Must divide evenly into the dataset sizes)')#0.0001
flags.DEFINE_string('gym_result_dir', 'gym-results/', 'Directory to put the gym results.')
flags.DEFINE_string('gym_env', 'CartPole-v0', 'Name of Open Gym\'s enviroment name. (CartPole-v0, CartPole-v1, MountainCar-v0)')
flags.DEFINE_boolean('step_verbose', False, 'verbose every step count')
flags.DEFINE_integer('step_verbose_count', 100, 'verbose step count')
flags.DEFINE_integer('save_step_count', 2000, 'model save step count')
flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'model save checkpoint_path (prefix is gym_env)')

FLAGS = flags.FLAGS

env = gym.make(FLAGS.gym_env)
#CC env = gym.wrappers.Monitor(env, directory=FLAGS.gym_env + "_" + FLAGS.gym_result_dir, force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Constants defining our neural network
config = Config(env, FLAGS.gym_env)
print(config.output_size)
print(type(config.output_size))

observation = env.reset() # CC
env.render() # CC
human_feedback = Feedback(env=env,key_type='1',h_up='3',h_down='2',h_right='1',h_left='-1',h_null='0') ## CC
#human_feedback = Feedback(env=env,key_type='1',h_up='1',h_down='-1',h_right='2',h_left='3',h_null='0') ## CC atari

renderdelay_f = True  # CC
humancontrol_f = False  # CC
oracle = False  # CC
f_RL = False # CC
input_size = (6,)
output_size = 5
alpha_1 = np.pi
loc = './CartPole-v0_f1_checkpoint/okish2T' # CC




def replay_train_h(mainH: PolicyEnsemble, targetH: PolicyEnsemble, train_batch: list, head: int) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (DeepQNetwork``): Main DQN that will be trained
        targetDQN (DeepQNetwork): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    #print(train_batch)
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
    e_fb = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])


    X = states
    y = np.zeros((len(X), mainH.output_size) )
    y[np.arange(len(X)), actions] = 1#np.ones((len(X),1))
    """for i in range(len(X)):
        if e_fb[i]==-1:
            y[i,:] = -1*(y[i,:]-1)"""
    y[e_fb==-1,:] = -1*(y[e_fb==-1,:]-1)
    # Train our network
    meanloss = mainH.update(X, y,head)
    # train error network
    """x_p = np.empty((0,len(states[0])))
    y_p = np.empty((0,1))
    for i in range(len(X)):
        if e_fb[i] != -1:
            act, un = mainH.predict(states[i])
            actionP = np.argmax(act)
            x_p = np.vstack((x_p,states[i]))
            if actions[i] == actionP:
                y_p = np.vstack((y_p,[0]))
            else:
                y_p = np.vstack((y_p, [1]))"""
    if np.random.rand() < 0.2:
        x_p = states[e_fb != -1]
        actionsNonNegFB = actions[e_fb != -1]
        y_p = np.zeros((len(x_p),1))
        act, u_e = mainH.predict(x_p)
        actionP = np.argmax(act,1)
        #print(len(y_p))
        if len(x_p)>0:
            y_p[(actionP-actionsNonNegFB)!=0,0] = 1
            residualloss = mainH.update_R(x_p,y_p)


    return meanloss

def replay_train(mainDQN: DeepQNetwork, targetDQN: DeepQNetwork, train_batch: list) -> float:
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (DeepQNetwork``): Main DQN that will be trained
        targetDQN (DeepQNetwork): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch[:FLAGS.batch_size]])
    rewards = np.array([x[2] for x in train_batch[:FLAGS.batch_size]])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch[:FLAGS.batch_size]])

    predict_result = targetDQN.predict(next_states)
    Q_target = rewards + FLAGS.discount_rate * np.max(predict_result, axis=1) * (1 - done)

    X = states
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target
    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def Qfeedback(u_a: float, u_e: float, th_u_a: float, th_u_e: float, action: np.ndarray, state: np.ndarray, teacherNet: DeepQNetwork, episode: int, step: int, ora_episodes: int, feedback_count: int) -> np.ndarray:
    mistakes = 0.0
    f_oracle = False
    correction = True
    e_fb = 0
    q_a = u_a > th_u_a
    q_e = u_e > th_u_e
    actionTeacher = np.argmax(teacherNet.predict(state))

    if ((feedback_count/(step+1))<0.1 or q_a or q_e ) and feedback_count<20 and  episode < ora_episodes: #corrections and evaluations
        if np.random.rand() < 0.5: #corrections
            if np.random.rand() < mistakes and not q_a:
                lottery = np.random.rand(teacherNet.output_size)
                lottery[actionTeacher] = 0
                actionTeacher = np.argmax(lottery)
        else:
            e_fb = 1 if (action == actionTeacher) else -1
            if np.random.rand() < mistakes and not q_a:
                e_fb = -1 * e_fb
            correction = False

        f_oracle = True

    return [ q_a, q_e, f_oracle, correction, actionTeacher, e_fb ]

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

def kukaee_call(data):
    global positionee
    positionee = data.data

def joint_call(data):
    global position, velocity
    position = data.position
    velocity = data.velocity

def joy_call(data):
    global h_joy
    h_joy = -1

    #if data.axes[3] > 0.9:
    #    h_joy = 0
    if data.axes[3] > 0.01:#data.axes[3] <= 0.9 and data.axes[3] > 0.01:
        h_joy = 1
    #if data.axes[3] < -0.9:
    #    h_joy = 4
    if data.axes[3] < -0.01:#data.axes[3] >= -0.9 and data.axes[3] < -0.01:
        h_joy = 3
    #if data.axes[3] >= -0.01 and data.axes[3] <= 0.01:
    #    h_joy = 2
    if data.axes[0] > 0.01:
        h_joy = 0
    if data.axes[0] < -0.01:
        h_joy = 4

    """if data.axes[5] > 0.9: #stop
        h_joy = -1"""
    if data.buttons[5] == 1: #pause episode while holding it
        h_joy = 10
    if data.buttons[4] == 1: #write buffer
        h_joy = 11

def body1_call(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
    global xB1, yB1
    xB1 = data.pose.position.x
    yB1 = data.pose.position.y - 0.0239

def body2_call(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
    global xB2, yB2
    xB2 = data.pose.position.x
    yB2 = data.pose.position.y

def body_call(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard pose %s", data.pose)
    global alpha, xB, yB
    xB = data.pose.position.x
    yB = data.pose.position.y
    x = data.pose.orientation.x
    y = data.pose.orientation.y
    z = data.pose.orientation.z
    w = data.pose.orientation.w
    roll, pitch, yaw = euler_from_quaternion(x, y, z, w)
    alpha = yaw + np.pi
    alpha = alpha - 2*np.pi if alpha > np.pi else alpha

def reset() -> np.ndarray:
    global count, posrequest, restart_flag, nomove, position
    count = 0
    delta = abs(0 - position[2])/30
    for itera in range(100):
        pose1.data = [position[2], 0.5, position[2], -np.pi / 2, 0, -0.5, 0]
        if 0-position[2] > (delta):
            pose1.data = [position[2] + delta, 0.5, position[2] + delta, -np.pi/2, 0, -0.5, 0]
        if 0-position[2] < -delta:
            pose1.data = [position[2] - delta, 0.5, position[2] - delta, -np.pi / 2, 0, -0.5, 0]
        pub_pos.publish(pose1)
        rospy.sleep(0.05)
    posrequest = 0
    pose1.data = [0, 0.5, 0, -np.pi / 2, 0, -0.50, 0]
    pub_pos.publish(pose1)
    rospy.sleep(1.0)
    state = get_state()
    restart_flag = True
    nomove = False
    return [state]

def get_state() -> np.ndarray:
    global alpha, alpha_1, xB, yB, positionee
    """l = 0.5755
    alpha = np.arccos(np.clip(((yB - positionee[2]) / l), -1.0, 1.0))
    alpha = alpha if (xB > positionee[1]) else -1 * alpha
    alpha_1 = alpha_1 + 2 * np.pi if (alpha > np.pi / 2 and alpha_1 < -np.pi / 2) else alpha_1
    alpha_1 = alpha_1 - 2 * np.pi if (alpha < -np.pi / 2 and alpha_1 > np.pi / 2) else alpha_1"""
    #alphap = alpha - alpha_1

    #alphap = np.clip(10*(alpha - alpha_1), -1,1)#so far the best
    alpha = xB - positionee[1]
    alphap = alpha -alpha_1
    #velocity = np.array(position) - np.array(position_1)
    state = np.array([alpha, alphap, position[0], velocity[0], position[2], velocity[2]])
    alpha_1 = alpha
    return state

def step(action: np.ndarray) -> np.ndarray:
    global  count, posrequest, yB1, restart_flag, nomove, posnomove
    poslimit = [-np.pi/12, np.pi/12] # xlow xup ylow yup zlow zup
    delta = 0.05

    if not nomove:
        posnomove = position[2]
    if action == 0:
        posrequest = position[2] - delta*1.5
        nomove = False
    if action == 1:  # CC
        posrequest = position[2] - delta/2
        nomove = False
    if action == 2:
        posrequest = posnomove
        nomove = True
    if action == 3:
        posrequest = position[2] + delta/2
        nomove = False
    if action == 4:  # CC
        posrequest = position[2] + delta*1.5
        nomove = False
    posrequest = poslimit[0] if posrequest < poslimit[0] else posrequest
    posrequest = poslimit[1] if posrequest > poslimit[1] else posrequest
    pose1.data = [posrequest, 0.5, posrequest, -np.pi/2, 0, -0.5, 0]

    state = get_state()

    while abs(state[0])>np.pi/3:
        #pose1.data = [0, np.pi / 2, 0, 0.6, 0.2, 0.6]
        #pub_pos.publish(pose1)
        rospy.sleep(0.05)
        #print(11111)
        state = get_state() #
        if abs(state[0])>np.pi/12:
            state[0] = np.pi/2
    pub_pos.publish(pose1)
    count = count + 1
    reward = abs(state[0])
    count = count + 1
    done = True if count > 2000 else False

    return [state, reward, done]


def main():
    global alpha, alpha_1, position, position_1, pub_pos, pose1, posrequest, xB, yB, fish_1, h_joy, count, positionee

    rospy.init_node('teleop', anonymous=True)
    rospy.Subscriber("iiwa/joint_states", JointState, joint_call)
    rospy.Subscriber("/vrpn_client_node/RigidBody3/pose", PoseStamped, body_call)
    rospy.Subscriber("/joy", Joy, joy_call)
    rospy.Subscriber("kuka_ee", Float64MultiArray, kukaee_call)
    pub_pos = rospy.Publisher("iiwa/PositionController/command", Float64MultiArray, queue_size=1)
    r = rospy.Rate(10)
    pose1 = Float64MultiArray()
    time.sleep(3)
    alpha_1 = xB - positionee[1]  # np.pi
    # pose1.data = [0, 0, 0, -np.pi / 2 +0.1, 0, 0, 0]
    # pub_pos.publish(pose1)
    # time.sleep(1)
    posrequest = 0  # position[0]
    reset()
    state = reset()


    renderdelay_f = False  # CC
    logger.info("FLAGS configure.")
    logger.info(FLAGS.__flags)

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=FLAGS.replay_memory_length)
    replay_buffer_h = deque(maxlen=FLAGS.replay_memory_length) # CC

    consecutive_len = 100 # default value
    if config.solving_criteria:
        consecutive_len = config.solving_criteria[0]
    last_n_game_reward = deque(maxlen=consecutive_len)

    sess = tf.Session()
    #teacherNet = StoredNetwork(loc, FLAGS.model_name, config.input_size, config.output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main2")

    mainDQN = DeepQNetwork(sess, FLAGS.model_name, input_size, output_size,
                           learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main")
    targetDQN = DeepQNetwork(sess,FLAGS.model_name, input_size, output_size, frame_size=FLAGS.frame_size, name="target")
    mainH = PolicyEnsemble(sess, "MLPv2Ensemble", input_size, output_size, learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="main_h")
    targetH = PolicyEnsemble(sess, "MLPv2Ensemble", input_size, output_size, frame_size=FLAGS.frame_size, name="target_h")

    sess.run(tf.global_variables_initializer())
    if oracle:
        teacherNet = DeepQNetwork(sess, FLAGS.model_name, input_size, output_size,
                               learning_rate=FLAGS.learning_rate, frame_size=FLAGS.frame_size, name="TeacherOracle")
        saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TeacherOracle'))
        saver2.restore(sess, loc)
    saver = tf.train.Saver(tf.global_variables())

    # initial copy q_net -> target_net
    copy_ops = get_copy_var_ops(dest_scope_name="target",
                                src_scope_name="main")
    copy_ops_h = get_copy_var_ops(dest_scope_name="target_h",
                                src_scope_name="main_h") # CC
    sess.run(copy_ops)
    sess.run(copy_ops_h)

    global_step = 1
    f_oracle = False
    ora_episodes = 50
    th_u_e = 0.075
    th_u_a = 0.3
    for episode in range(FLAGS.max_episode_count):
        e = 1. / (((episode ) / 20) + 1)#1. / ((episode / 10) + 1)
        eO = 1 #1. / ((episode / 10) + 2)
        eO2 = 1.75 #1. / (((episode-50) / 30) + 1.007)
        done = False
        step_count = 0
        feedback_count = 0
        certain_actions = 0
        e_fb = 1
        e_reward = 0
        model_loss = 0
        avg_reward = np.mean(last_n_game_reward)
        actionvar_e =[]
        actionvar_a =[]

        state = reset()
        print('Episode started')
        while not done:
            h = human_feedback.get_h() # CC Get feedback signal
            actions, u_e = mainH.predict(state)
            action = np.argmax(actions)
            u_a = mainH.predict_R(state)
            #print(state)
            actionvar_e.append(u_e)
            actionvar_a.append(u_a)
            if oracle:  # with humans, the evaluative FB goes after the action execution, with oracles it does not matter
                q_a, q_e, f_oracle, correction, actionTeacher, e_fb = Qfeedback(u_a,u_e,th_u_a,th_u_e,action,state,teacherNet,episode,step_count,ora_episodes,feedback_count)
                action = actionTeacher if (f_oracle and correction) else action
                certain_actions = certain_actions+1 if (not f_oracle) and (u_e<th_u_e) else certain_actions
                feedback_count = feedback_count+1 if f_oracle else feedback_count

            if f_RL and u_e>th_u_e:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

            if renderdelay_f and step_count == 1: # CC
                time.sleep(0.005) # CC
            if h[0] == 4:
                renderdelay_f = True
            if h[0] == 5:
                renderdelay_f = False

            if h_joy == 0:
                action = 0
            if h_joy == 1:
                action = 1
            if h_joy == 2:
                action = 2
            if h_joy == 3:
                action = 3
            if h_joy == 4:
                action = 4
            if h_joy == 10:
                action = 2
                count = count - 1
            env.render() # CC
            if renderdelay_f: # CC
                time.sleep(0.001) # CC

            # Get new state and reward from environment
            next_state, reward, done = step(action)

            """if oracle and np.random.rand() < eO and episode < ora_episodes and step_count > 3 and step_count < 7:  # sometimes choose an action of the teacher
                if np.argmax(teacherNet.predict(state)) == action:
                    e_fb = 1
                else:
                    e_fb = -1
                f_oracle = True"""
            # Save the experience to our buffer
            if episode > ora_episodes and u_e>=th_u_e and not f_oracle and f_RL:
                replay_buffer.append((state, action, reward, next_state, done))
            if h[0] == 1 or h[0] == -1 or (h_joy != -1 and h_joy != 10 and h_joy != 11):
                replay_buffer_h.append((state, action, e_fb))
            if f_oracle:
                replay_buffer_h.append((state, action, e_fb))
                f_oracle = False

            if len(replay_buffer) > FLAGS.batch_size and f_RL:
                minibatch = random.sample(replay_buffer, (FLAGS.batch_size))
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                model_loss = loss
            """if len(replay_buffer_h) > FLAGS.batch_size:# and episode < ora_episodes: # CC
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch,0)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch,1)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch,2)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch,3)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch,4)"""
                #minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                #loss, _ = replay_train(mainDQN, targetDQN, minibatch)#CC
                #model_loss = loss#CC
            if h_joy == 11:
                file_buffer = open('./pendulum_buffers/pendulum_buffer2.rec', 'wb')
                pickle.dump(replay_buffer_h, file_buffer)

            if step_count % FLAGS.target_update_count == 0:
                sess.run(copy_ops)
                sess.run(copy_ops_h)

            state = next_state
            e_reward += reward
            step_count += 1

            # save model checkpoint
            if global_step % FLAGS.save_step_count == 0:
                checkpoint_path = FLAGS.gym_env + "_f" + str(FLAGS.frame_size) + "_" + FLAGS.checkpoint_path + "global_step"
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                saver.save(sess, checkpoint_path, global_step=global_step)
                logger.info("save model for global_step: {global_step} ")

            global_step += 1
            r.sleep()
            if rospy.is_shutdown():
                print('shutdown')
                break
        print('Episode finished')
        print(len(replay_buffer_h))
        print(step_count)
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

        #if episode > ora_episodes:
        logger.info("mean var: "+str(np.mean(actionvar_e))+" min: "+str(np.min(actionvar_e))+" max: "+str(np.max(actionvar_e)))
        logger.info("mean Avar: " + str(np.mean(actionvar_a)) + " Amin: " + str(np.min(actionvar_a)) + " Amax: " + str(
            np.max(actionvar_a)))
        logger.info("certain actions ratio: "+str(certain_actions/step_count))

        if len(replay_buffer_h) > FLAGS.batch_size:
            ne = 100
            if episode > 5:
                n5 = 5000
            for k in range(ne): #10000
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch, 0)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch, 1)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch, 2)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch, 3)
                minibatch = random.sample(replay_buffer_h, (FLAGS.batch_size))
                loss = replay_train_h(mainH, targetH, minibatch, 4)
                #print(loss)


        if len(last_n_game_reward) == last_n_game_reward.maxlen:
            avg_reward = np.mean(last_n_game_reward)

            if config.solving_criteria and  avg_reward > (config.solving_criteria[1]):
                logger.info("Game Cleared in {episode} episodes with avg reward {avg_reward}")
                break

        if rospy.is_shutdown():
            print('shutdown')
            break



if __name__ == "__main__":
    if FLAGS.model_name.startswith("MLP") and FLAGS.frame_size > 1:
        raise ValueError('do not support frame_size > 1 if model_name is MLP')

    main()

