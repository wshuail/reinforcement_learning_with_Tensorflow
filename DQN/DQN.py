#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DQN(object):
    def __init__(self, n_states, n_actions, learning_rate=0.0003, epsilon=1,
                 epsilon_min=0.001, epsilon_decay_step=300, gamma=0.9, batch_size=50):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.gamma = gamma
        self.batch_size = batch_size

        self.learning_step = 0

        self.memory_size = 5000
        self.memory_counter = 0
        self.memory = np.zeros([self.memory_size, self.n_states * 2 + 3])

        self._build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.loss_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.q_target = tf.placeholder(tf.float32, [None], name='q_target')

        k_init, b_init = tf.random_uniform_initializer(0., 0.2), tf.constant_initializer(0.1)

        hl_1 = tf.layers.dense(inputs=self.s, units=64,
                               activation=tf.nn.relu,
                               kernel_initializer=k_init,
                               bias_initializer=b_init,
                               name='hl_1')
        self.q_eval = tf.layers.dense(inputs=hl_1, units=self.n_actions,
                                      activation=None,
                                      kernel_initializer=k_init,
                                      bias_initializer=b_init,
                                      name='q_eval')

        actions_reward = tf.reduce_sum(tf.multiply(self.q_eval, tf.one_hot(indices=self.a, depth=self.n_actions)), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.q_target - actions_reward))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            action_values = self.session.run(self.q_eval, feed_dict={self.s: s})
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
        if self.memory_counter <= self.memory_size:
            index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, :self.n_states]
        ba = batch_memory[:, self.n_states]
        br = batch_memory[:, self.n_states+1]
        bs_ = batch_memory[:, -(self.n_states+1):-1]
        bt = batch_memory[:, -1]

        q_next = self.session.run(self.q_eval, feed_dict={self.s: bs_})
        q_next = np.max(q_next, axis=1)
        q_target = br + self.gamma*q_next*bt

        _, loss = self.session.run([self.train_op, self.loss],
                                   feed_dict={self.s: bs,
                                              self.a: ba,
                                              self.q_target: q_target})

        self.loss_his.append(loss)

        self.learning_step += 1
        self.epsilon -= (self.epsilon - self.epsilon_min)/self.epsilon_decay_step

    def plot_result(self, data, x_label, y_label):
        plt.plot(np.arange(len(data)), data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
