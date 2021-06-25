import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf # irene
from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from tabulate import tabulate

"""
D-COACH implementation
"""


class DCOACH:
    def __init__(self, dim_a, dim_o, action_upper_limits, action_lower_limits, e, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode, policy_model_learning_rate, human_model_learning_rate, human_model_included):
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
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        self.human_model_included = human_model_included
        self.e =  float(e)
        self.buffer = Buffer(min_size=self.buffer_min_size, max_size=self.buffer_max_size)






    def createModels(self, neural_network):
            print('CREATE MODELS')
            self.policy_model = neural_network.policy_model()
            if self.human_model_included:
                self.Human_model = neural_network.Human_model()


    def _generate_policy_label(self, action):
        if np.any(self.h):

            error = [self.h[0]*self.e, self.h[1]*self.e, self.h[2]*self.e]

            error = np.array(error).reshape(1, self.dim_a)
            self.policy_action_label = []

            for i in range(self.dim_a):
                self.policy_action_label.append(np.clip(action[i] / self.action_upper_limits[i] + error[0, i], -1, 1))

            self.policy_action_label = np.array(self.policy_action_label).reshape(1, self.dim_a)
        else:

            self.policy_action_label = np.reshape(action, [1, self.dim_a])


    def _generate_batch_policy_label(self, action_batch, h_predicted_batch):


        if np.any(h_predicted_batch):

            multi = np.asarray(h_predicted_batch) * self.e
            error = multi.reshape(8, self.dim_a)

            a_target_batch = []

            # a_target = a + error
            # numpy.clip(a, a_min, a_max)
            for i in range(8):
                a_target_batch.append(np.clip(action_batch[i] / self.action_upper_limits[0] + error[i], -1, 1))

            a_target_batch = np.array(a_target_batch).reshape(8, self.dim_a)

        else:
            a_target_batch = np.reshape(action_batch, [8, self.dim_a])

        return a_target_batch

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

            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))








    def _batch_update(self, neural_network, batch, i_episode, t):
        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_batch_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]),
                                                                  dtype=tf.float32)
        action_label_batch = [np.array(pair[1]) for pair in batch]


        self._single_update(neural_network, observations_batch_reshaped_tensor, action_label_batch)


    def Human_single_update(self, observation, action, h_human):




        # TRAIN Human model
        optimizer_Human_model = tf.keras.optimizers.Adam(learning_rate=self.human_model_learning_rate)

        with tf.GradientTape() as tape_policy:


            h_predicted = self.Human_model([observation, action])



            policy_loss = 0.5 * tf.reduce_mean(tf.square(h_predicted - h_human))

            grads = tape_policy.gradient(policy_loss, self.Human_model.trainable_variables)



        optimizer_Human_model.apply_gradients(zip(grads, self.Human_model.trainable_variables))

    def Human_batch_update(self, batch):
        #print('bufferF batch update')

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  # last



        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [8, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [8, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [8, 1]), dtype=tf.float32)


        self.Human_single_update(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)




    def feed_h(self, h):
        self.h = h

        if np.any(self.h) and self.human_model_included:
            self.h_to_buffer = tf.convert_to_tensor(np.reshape(self.h[0], [1, 1]), dtype=tf.float32)





    def action(self, state_representation):
        self.state_representation = state_representation
        action = self.policy_model([self.state_representation])


        if self.human_model_included:
            self.action_to_buffer = action
        action = action.numpy()


        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def TRAIN_Human_Model_NOT_included(self, neural_network, action, t, done, i_episode):
        self._generate_policy_label(action)
        # print('train agent')
        # Policy training
        if np.any(self.h):  # if any element is not 0
            print('train!')
            self._single_update(neural_network, self.state_representation, self.policy_action_label)
            ###print('agent single update')
            ###print("feedback:", self.h)

            # Add last step to memory buffer
            if self.policy_action_label is not None:
                self.buffer.add([self.state_representation, self.policy_action_label])
            # Add last step to memory buffer
            # if transition_model.last_step(self.policy_action_label) is not None:
            # self.buffer.add(transition_model.last_step(self.policy_action_label))

            # Train sampling from buffer
            if self.buffer.initialized():
                ###print('Train sampling from buffer')

                batch = self.buffer.sample(
                    batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                self._batch_update(neural_network, batch, i_episode, t)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            print('Train policy every k time steps from buffer')
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(neural_network, batch, i_episode, t)

    # If feedback is received: Do main loop
    def TRAIN_Human_Model_included(self, neural_network, action, t, done, i_episode):

        if np.any(self.h):  # if any element is not 0
            #print(self.h)
            # 8. Append last step to buffer
            #print('TRAIN')
            print('train!')
            self.buffer.add([self.state_representation, self.action_to_buffer, self.h_to_buffer])

            # 1. Generate a_target
            self._generate_policy_label(action)

            # 2. Update policy with current observation and a_target
            #print('agent single update')
            self._single_update(neural_network, self.state_representation, self.policy_action_label)

            # 3. Update Human model with a minibatch sampled from buffer D
            if self.buffer.initialized():
                #print('Train sampling from buffer')
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                #self._batch_update(neural_network, batch, i_episode, t)
                self.Human_batch_update(batch)

                # 4. Batch of observations from buffer --> Policy --> Batch of actions

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
                observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]), dtype=tf.float32)


                optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)

                with tf.GradientTape() as tape_policy:

                    #policy_output = self.policy_model([state_representation])
                    actions_batch = self.policy_model([observations_reshaped_tensor])

                    # 5. Get bath of h predictions from Human model
                    h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

                    # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
                    a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

                    # 7. Update policy indirectly from Human model
                    policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
                    grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

                optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))






        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            #print('Train policy every k time steps from buffer')

            # update Human model
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self.Human_batch_update(batch)

            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
            observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [8, self.dim_o]),
                                                                dtype=tf.float32)
            # update Policy model
            optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.policy_model_learning_rate)

            with tf.GradientTape() as tape_policy:
                # policy_output = self.policy_model([state_representation])
                actions_batch = self.policy_model([observations_reshaped_tensor])

                # 5. Get bath of h predictions from Human model
                h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

                '''
                h_predicted_discretized = []
                for i in range(len(h_predicted_batch)):
                    if h_predicted_batch[i] < 0:
                        h_predicted_discretized.append(-1)
                    else:
                        h_predicted_discretized.append(1)               
                '''

                # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
                a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

                # 7. Update policy indirectly from Human model
                policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
                grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

            optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))
