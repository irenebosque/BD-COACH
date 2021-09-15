import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf # irene
# from metaworld.policies.action import Action
# from metaworld.policies.policy import Policy, assert_fully_parsed, move
from tabulate import tabulate
import pandas as pd

"""
D-COACH implementation
"""


class DCOACH:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, e, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, human_model_learning_rate, human_model_included, action_limit, agent_with_hm_learning_rate):
        # Initialize variables
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        #self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.dim_o = dim_o
        self.action_upper_limits = [1,1,1,1]
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.train_end_episode = train_end_episode
        self.policy_model_learning_rate = policy_model_learning_rate
        self.human_model_learning_rate = human_model_learning_rate
        self.agent_with_hm_learning_rate = agent_with_hm_learning_rate
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.human_model_included = human_model_included
        self.e =  float(e)
        self.action_limit = float(action_limit)
        self.buffer = Buffer(min_size=self.buffer_min_size, max_size=self.buffer_max_size)
        self.discretize_t = 0.05
        self.policy_loss_list = []
        self.policy_loss_hm_batch = 0
        self.policy_loss_agent_batch = 0






    def createModels(self, neural_network):
            print('CREATE MODELS')
            self.policy_model = neural_network.policy_model()
            if self.human_model_included:
                self.Human_model = neural_network.Human_model()



    def feed_h(self, h):
        print('\n')

        if np.any(h):
            self.h = h
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(self.h, [1, self.dim_a]), dtype=tf.float32) # When Human model = True

            print('Yes feedback', self.h)


        else:
            print('No feedback')
        # self.h = h
        #
        # if np.any(self.h) and self.human_model_included:
        #     self.h_to_buffer = tf.convert_to_tensor(np.reshape(self.h, [1, self.dim_a]), dtype=tf.float32)
        #     print('feed_h, self.h_to_buffer', self.h_to_buffer)



    def _generate_policy_label(self, action, h):



        error = [0] * self.dim_a
        for i, name in enumerate(h):
            error[i] = h[i]*self.e



        #error = [h[0]*self.e, h[1]*self.e, h[2]*self.e]

        error = np.array(error).reshape(1, self.dim_a)
        policy_action_label = []

        for i in range(self.dim_a):
            policy_action_label.append(np.clip(action[i] / self.action_limit + error[0, i], -1, 1))

        policy_action_label = np.array(policy_action_label).reshape(1, self.dim_a)

        return policy_action_label
        # else:
        #
        #     self.policy_action_label = np.reshape(action, [1, self.dim_a])


    def _generate_batch_policy_label(self, action_batch, h_predicted_batch):


        if np.any(h_predicted_batch):

            multi = np.asarray(h_predicted_batch) * self.e

            # print('h_predicted_batch',  h_predicted_batch)
            error = multi.reshape(8, self.dim_a)


            a_target_batch = []

            # a_target = a + error
            # numpy.clip(a, a_min, a_max)
            for i in range(8):
                a_target_batch.append(np.clip(action_batch[i] / self.action_limit + error[i], -1, 1))

            a_target_batch = np.array(a_target_batch).reshape(8, self.dim_a)

            return a_target_batch
        #
        # else:
        #     a_target_batch = np.reshape(action_batch, [8, self.dim_a])

       # return a_target_batch

    def _single_update(self, neural_network, state_representation, policy_label):

        # TRAIN policy model
        optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)

        with tf.GradientTape() as tape_policy:

            policy_output = self.policy_model([state_representation])
            '''
            print("\n")
            

            data = [["action^*", policy_label[0][0], policy_label[0][1], policy_label[0][2], policy_label[0][3]],
                    ["action_agent", policy_output[0][0], policy_output[0][1], policy_output[0][2], policy_output[0][3]]]

            print(tabulate(data, headers=["what", "dx", "dy", "dz", "gripper"]))
            '''

            # print("policy_output", policy_output)
            # print("policy_label", policy_label)
            # dif = policy_output - policy_label
            # print("dif",dif)

            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))

            policy_loss_float = policy_loss.numpy()
            self.policy_loss_agent = policy_loss_float

            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))
        return self.policy_loss_agent








    def _batch_update(self, neural_network, batch, i_episode, t):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_batch_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]),
                                                                  dtype=tf.float32)
        action_label_batch = [np.array(pair[1]) for pair in batch]
        #print(" action_label_batch: ",  action_label_batch)
        action_label_batch  = tf.convert_to_tensor(np.reshape(action_label_batch , [8, self.dim_a]), dtype=tf.float32)


        # print('\n')
        # print('obs_hand', observations_batch[0][0])
        #
        # print('action_label_batch', action_label_batch[0])
        # print('\n')




        self.policy_loss_agent_batch =  self._single_update(neural_network, observations_batch_reshaped_tensor, action_label_batch)


    def Human_single_update(self, observation, action, h_human):




        # TRAIN Human model
        optimizer_Human_model = tf.keras.optimizers.Adam(learning_rate=self.human_model_learning_rate)

        with tf.GradientTape() as tape_policy:


            h_predicted = self.Human_model([observation, action])
            # print('Human_single_update, h_predicted before %%%%%%%%%%%%%%%%', h_predicted)
            # h_predicted = tf.where(tf.math.greater(h_predicted, self.discretize_t), 1, h_predicted)
            # h_predicted = tf.where(tf.math.less(h_predicted, -1*self.discretize_t), -1, h_predicted)
            # h_predicted = tf.where(tf.logical_and(tf.math.less_equal(h_predicted, self.discretize_t),
            #                                             tf.math.greater_equal(h_predicted, -1*self.discretize_t)), 0,
            #                              h_predicted)


            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            # print('Human_single_update, h_predicted after%%%%%%%%%%%%%%%%', h_predicted)
            # print('Human_single_update, h_human%%%%%%%%%%%%%%%%', h_human)

            # print("h_human", h_human)
            # print("h_predicted", h_predicted)
            # dif = h_human-h_predicted
            # print("dif", dif)

            policy_loss = 0.5 * tf.reduce_mean(tf.square(h_human-h_predicted ))
            policy_loss_float = policy_loss.numpy()
            self.policy_loss_hm = policy_loss_float

            grads = tape_policy.gradient(policy_loss, self.Human_model.trainable_variables)



        optimizer_Human_model.apply_gradients(zip(grads, self.Human_model.trainable_variables))
        return self.policy_loss_hm

    def Human_batch_update(self, batch):
        #print('bufferF batch update')

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  # last


        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [8, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [8, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [8, self.dim_a]), dtype=tf.float32)



        self.policy_loss_hm_batch = self.Human_single_update(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)
        #print("self.policy_loss_hm_batch", self.policy_loss_hm_batch)









    def action(self, state_representation):
        #self.observation = state_representation

        #state_representation = np.reshape(state_representation [1, self.dim_o])
        # print('state representation', state_representation)
        # print('state representation2', [state_representation])
        action = self.policy_model(state_representation)


        if self.human_model_included:
            self.action_to_buffer = action
        action = action.numpy()


        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_limit
            out_action.append(action[0, i])


        return np.array(out_action)

    # def TRAIN_Human_Model_NOT_included(self, neural_network, action, t, done, i_episode, h):
    #
    #     if np.any(h):  # if any element is not 0
    #
    #
    #         for i in range(len(self.buffer.buffer)):
    #             print(self.buffer.buffer[i])
    #
    #
    #         self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]),
    #                                                 dtype=tf.float32)  # When Human model = True
    #
    #         action_label = self._generate_policy_label(action, h)
    #
    #         self._single_update(neural_network, self.observation, action_label)
    #
    #
    #         # Add last step to memory buffer
    #         print('action_label: ',action_label)
    #         if action_label is not None:
    #
    #             self.buffer.add([self.observation, action_label])
    #
    #
    #         # Train sampling from buffer
    #         if self.buffer.initialized():
    #             ###print('Train sampling from buffer')
    #
    #             batch = self.buffer.sample(
    #                 batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
    #             self._batch_update(neural_network, batch, i_episode, t)
    #
    #     # Train policy every k time steps from buffer
    #     if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
    #         #print('Train policy every k time steps from buffer')
    #         batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
    #         self._batch_update(neural_network, batch, i_episode, t)

    # If feedback is received: Do main loop
    def TRAIN_Human_Model_included(self, neural_network, action, t, done, i_episode, h, observation, cummulative_feedback, t_total):
        #print("cummulative_feedback: ", cummulative_feedback)

        if np.any(h):  # if any element is not 0


            # 1. append  (o_t, a_t, h_t) to D

            self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]),
                                                    dtype=tf.float32)  # When Human model = True

            self.buffer.add([observation, self.action_to_buffer, self.h_to_buffer])
            # for i in range(len(self.buffer.buffer)):
            #     print("buffer", self.buffer.buffer[i])





            # 2. Generate a_target
            action_label = self._generate_policy_label(action, h)

            # 3. Update policy with current observation and a_target
            self._single_update(neural_network, observation, action_label)

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                if t_total < 40000:
                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                    self.Human_batch_update(batch)

                # 4. Batch of observations from buffer --> Policy --> Batch of actions

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
                observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]), dtype=tf.float32)

                if cummulative_feedback > 100:
                    #print("Update policy with HM 1")
                    #optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)
                    optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.agent_with_hm_learning_rate)

                    with tf.GradientTape() as tape_policy:

                        #policy_output = self.policy_model([state_representation])
                        actions_batch = self.policy_model([observations_reshaped_tensor])

                        # 5. Get bath of h predictions from Human model
                        h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])


                        h_predicted_batch= h_predicted_batch.numpy()


                        h_predicted_batch = h_predicted_batch.tolist()


                        # for n, i in enumerate(h_predicted_batch):
                        #


                        # for i in range(len(h_predicted_batch)):
                        #     for j in range(len(h_predicted_batch[i])):
                        #         if h_predicted_batch[i][j] > -1 and h_predicted_batch[i][j] < -0.1:
                        #             h_predicted_batch[i][j] = -1
                        #         elif h_predicted_batch[i][j] > 0.1 and h_predicted_batch[i][j] < 1:
                        #             h_predicted_batch[i][j] = 1
                        #         else:
                        #             h_predicted_batch[i][j] = 0










                        # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
                        a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

                        # 7. Update policy indirectly from Human model


                        policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
                        #print("policy_loss", policy_loss)
                        grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)
                        self.policy_loss_agent_batch = policy_loss

                    optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))





# for i in range(len(h_predicted_batch)):
                    #     for j in range(len(h_predicted_batch[i])):
                    #         if h_predicted_batch[i][j] > -1 and h_predicted_batch[i][j] < -0.1:
                    #             h_predicted_batch[i][j] = -1
                    #         elif h_predicted_batch[i][j] > 0.1 and h_predicted_batch[i][j] < 1:
                    #             h_predicted_batch[i][j] = 1
                    #         else:
                    #             h_predicted_batch[i][j] = 0
        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            #print('Train policy every k time steps from buffer')

            # update Human model
            if t_total < 40000:
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.Human_batch_update(batch)

            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
            observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]),
                                                                dtype=tf.float32)
            if cummulative_feedback > 100:
                #print("Update policy with HM 2")
                # update Policy model
                #optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)
                optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.agent_with_hm_learning_rate)

                with tf.GradientTape() as tape_policy:
                    # policy_output = self.policy_model([state_representation])
                    actions_batch = self.policy_model([observations_reshaped_tensor])

                    # 5. Get bath of h predictions from Human model
                    h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

                    h_predicted_batch = h_predicted_batch.numpy()

                    h_predicted_batch = h_predicted_batch.tolist()


                    # for i in range(len(h_predicted_batch)):
                    #     for j in range(len(h_predicted_batch[i])):
                    #         if h_predicted_batch[i][j] > -1 and h_predicted_batch[i][j] < -0.1:
                    #             h_predicted_batch[i][j] = -1
                    #         elif h_predicted_batch[i][j] > 0.1 and h_predicted_batch[i][j] < 1:
                    #             h_predicted_batch[i][j] = 1
                    #         else:
                    #             h_predicted_batch[i][j] = 0



                    # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
                    a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

                    # print("A_TARGET_BATCH", a_target_batch)
                    # print("actions_batch", actions_batch)
                    # dif = actions_batch - a_target_batch
                    # print("dif", dif)

                    # 7. Update policy indirectly from Human model
                    policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
                    grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

                optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

    def TRAIN_Human_Model_NOT_included(self, neural_network, action, t, done, i_episode, h, observation):

        if np.any(h):  # if any element is not 0
            print('TRAINN')

            if h[0] == 10: # Button A pressed to reduce speed
                print("Slow down!")
                #print("action: ", action)

                action_label = [action*0.5]
                #print('label: ', action_label)
            else:

                # 2. Generate a_target
                action_label = self._generate_policy_label(action, h)
                #print('label2: ', action_label)
            

            self.buffer.add([observation, action_label])


            # for i in range(len(self.buffer.buffer)):
            #     print("buffer", self.buffer.buffer[i])




            # 3. Update policy with current observation and a_target
            self._single_update(neural_network, observation, action_label)

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                self._batch_update(neural_network, batch, i_episode, t)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(neural_network, batch, i_episode, t)
