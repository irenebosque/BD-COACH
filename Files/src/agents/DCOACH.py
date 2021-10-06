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
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, human_model_learning_rate, human_model_included, action_limit, agent_with_hm_learning_rate, h_threshold):
        # Initialize variables
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        #self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.dim_o = dim_o
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
        self.policy_loss_list = []
        self.h_threshold = float(h_threshold)





    def createModels(self, neural_network):
            print('CREATE MODELS')
            self.policy_model = neural_network.policy_model()
            if self.human_model_included:
                self.Human_model = neural_network.Human_model()

    def discretize_feedback(self, h_predicted_batch):

        h_predicted_batch = h_predicted_batch.numpy()
        h_predicted_batch = h_predicted_batch.tolist()

        for i in range(len(h_predicted_batch)):
            for j in range(len(h_predicted_batch[i])):
                if h_predicted_batch[i][j] > -1 and h_predicted_batch[i][j] < -1 * self.h_threshold:
                    h_predicted_batch[i][j] = -1
                elif h_predicted_batch[i][j] > self.h_threshold and h_predicted_batch[i][j] < 1:
                    h_predicted_batch[i][j] = 1
                else:
                    h_predicted_batch[i][j] = 0



        return h_predicted_batch




    # def feed_h(self, h):
    #     print('\n')
    #
    #     if np.any(h):
    #         self.h = h
    #         self.h_to_buffer = tf.convert_to_tensor(np.reshape(self.h, [1, self.dim_a]), dtype=tf.float32) # When Human model = True
    #
    #         print('Yes feedback', self.h)
    #
    #
    #     else:
    #         print('No feedback')




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


        #if np.any(h_predicted_batch):


        multi = np.asarray(h_predicted_batch) * self.e

        # print('h_predicted_batch',  h_predicted_batch)
        error = multi.reshape(self.buffer_sampling_size, self.dim_a)


        a_target_batch = []

        # a_target = a + error
        # numpy.clip(a, a_min, a_max)
        for i in range(self.buffer_sampling_size):

            a_target_batch.append(np.clip(action_batch[i] / self.action_limit + error[i], -1, 1))

        a_target_batch = np.array(a_target_batch).reshape(self.buffer_sampling_size, self.dim_a)


        return a_target_batch


    def _single_update(self, state_representation, policy_label):

        # TRAIN policy model
        optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)

        with tf.GradientTape() as tape_policy:

            policy_output = self.policy_model([state_representation])

            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))



            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        return








    def _batch_update(self, batch, i_episode, t):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_batch_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                                  dtype=tf.float32)
        action_label_batch = [np.array(pair[1]) for pair in batch]
        #print(" action_label_batch: ",  action_label_batch)
        action_label_batch  = tf.convert_to_tensor(np.reshape(action_label_batch , [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)





        self._single_update(observations_batch_reshaped_tensor, action_label_batch)

    def _policy_batch_update_with_HM(self, batch):


        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                            dtype=tf.float32)

        optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.agent_with_hm_learning_rate)

        with tf.GradientTape() as tape_policy:
            # policy_output = self.policy_model([state_representation])
            actions_batch = self.policy_model([observations_reshaped_tensor])

            # 5. Get bath of h predictions from Human model
            h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

            h_predicted_batch = self.discretize_feedback(h_predicted_batch)

            # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
            #print("actions_batch: ", actions_batch)
            #print("h_predicted_batch: ", h_predicted_batch)
            a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

            # 7. Update policy indirectly from Human model

            #print("a_target_batch", a_target_batch)

            policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))

    def Human_single_update(self, observation, action, h_human):


        # TRAIN Human model
        optimizer_Human_model = tf.keras.optimizers.Adam(learning_rate=self.human_model_learning_rate)

        with tf.GradientTape() as tape_policy:

            h_predicted = self.Human_model([observation, action])
            #print("h_human: ", h_human)
            #print("h_predicted: ", h_predicted)
            policy_loss = 0.5 * tf.reduce_mean(tf.square(h_human- h_predicted))
            grads = tape_policy.gradient(policy_loss, self.Human_model.trainable_variables)

        optimizer_Human_model.apply_gradients(zip(grads, self.Human_model.trainable_variables))

        return


    def Human_batch_update(self, batch):
        #print('bufferF batch update')

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  # last
        #print("h_human_batch: ",h_human_batch)


        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [self.buffer_sampling_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [self.buffer_sampling_size, self.dim_a]), dtype=tf.float32)



        self.Human_single_update(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)







    def action(self, state_representation):

        action = self.policy_model(state_representation)



        if self.human_model_included:
            self.action_to_buffer = action


        action = action.numpy()


        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_limit
            out_action.append(action[0, i])




        return np.array(out_action)


    def TRAIN_Human_Model_included(self, action, h, observation, t, done):





        if np.any(h):  # if any element is not 0


            # 1. append  (o_t, a_t, h_t) to D
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)


            self.buffer.add([observation, self.action_to_buffer, self.h_to_buffer])


            # 2. Generate a_target
            action_label = self._generate_policy_label(action, h)

            # 3. Update policy with current observation and a_target
            self._single_update(observation, action_label)

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.Human_batch_update(batch)

                # 4. Batch update of the policy with the Human Model

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._policy_batch_update_with_HM(batch)






        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            #print('Train policy every k time steps from buffer')

            # update Human model
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self.Human_batch_update(batch)

            # Batch update of the policy with the Human Model
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._policy_batch_update_with_HM(batch)









    def TRAIN_Human_Model_NOT_included(self, action, t, done, i_episode, h, observation):

        if np.any(h):  # if any element is not 0




            # 2. Generate a_target
            action_label = self._generate_policy_label(action, h)


            self.buffer.add([observation, action_label])


            # 3. Update policy with current observation and a_target

            self._single_update(observation, action_label)

            # 4. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                self._batch_update(batch, i_episode, t)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(batch, i_episode, t)


