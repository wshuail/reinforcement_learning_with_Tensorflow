#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DoubleDQN(object):
    def __init__(self, n_states, n_actions, double_dqn=False, learning_rate=0.002,
                 epsilon=1, epsilon_min=0.001, epsilon_decay_step=300,
                 gamma=0.95, batch_size=50):
        self.n_states = n_states
        self.n_actions = n_actions
        self.double_dqn = double_dqn
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.gamma = gamma
        self.batch_size = batch_size

        self.learning_step = 0
        self.params_replace_step = 50

        self.memory_size = 5000
        self.memory_counter = 0
        self.memory = np.zeros([self.memory_size, self.n_states * 2 + 3])

        self._build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.loss_his = []

    def _build_net(self):

        k_init, b_init = tf.random_uniform_initializer(0., 0.2), tf.constant_initializer(0.1)

        # build eval net
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name='s_')
        self.q_target = tf.placeholder(tf.float32, [None], name='q_target')

        with tf.variable_scope('eval_net'):
            hidden_layer = tf.layers.dense(inputs=self.s, units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=k_init,
                                           bias_initializer=b_init,
                                           name='hidden_layer')
            self.eval_net = tf.layers.dense(inputs=hidden_layer, units=self.n_actions,
                                            activation=None,
                                            kernel_initializer=k_init,
                                            bias_initializer=b_init,
                                            name='eval_net')

        # build target net
        with tf.variable_scope('target_net'):
            hidden_layer = tf.layers.dense(inputs=self.s_, units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=k_init,
                                           bias_initializer=b_init,
                                           name='hidden_layer')
            self.target_net = tf.layers.dense(inputs=hidden_layer, units=self.n_actions,
                                              activation=None,
                                              kernel_initializer=k_init,
                                              bias_initializer=b_init,
                                              name='target_net')

        with tf.variable_scope('loss'):
            actions_reward = tf.reduce_sum(tf.multiply(self.eval_net, tf.one_hot(indices=self.a, depth=self.n_actions)),
                                           reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.q_target - actions_reward))
        with tf.variable_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def _update_params(self):
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        update_params_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # update target net params
        self.session.run(update_params_op)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            action_values = self.session.run(self.eval_net, feed_dict={self.s: s})
            action = np.argmax(action_values)
        else:
            action = np.random.choice(range(self.n_actions))
        return action

    def store_transition(self, s, a, r, s_, terminal):
        transition = np.hstack((s, a, r, s_, terminal))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learning_step % self.params_replace_step == 0:
            self._update_params()

        if self.memory_counter <= self.memory_size:
            index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, :self.n_states]
        ba = batch_memory[:, self.n_states]
        br = batch_memory[:, self.n_states + 1]
        bs_ = batch_memory[:, -(self.n_states+1):-1]
        bt = batch_memory[:, -1]

        q_target_next = self.session.run(self.target_net, feed_dict={self.s_: bs_})

        if self.double_dqn:
            q_eval_next = self.session.run(self.eval_net, feed_dict={self.s: bs_})
            a4next = np.argmax(q_eval_next, axis=1)
            q_target = br + self.gamma * q_target_next[np.arange(self.batch_size, dtype=np.int32), a4next]*bt
        else:
            q_next = np.max(q_target_next, axis=1)
            q_target = br + self.gamma*q_next*bt

        _, loss = self.session.run([self.train_op, self.loss],
                                   feed_dict={self.s: bs,
                                              self.a: ba,
                                              self.q_target: q_target})

        self.loss_his.append(loss)

        self.learning_step += 1
        self.epsilon = (self.epsilon - self.epsilon_min)/self.epsilon_decay_step

    def plot_result(self, data, x_label, y_label):
        plt.plot(np.arange(len(data)), data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
