#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class DDQN(object):
    def __init__(self, n_states, n_actions, gamma=0.95, learning_rate=0.005,
                 epsilon=1, epsilon_min=0.001, epsilon_decay_step=300, memory_capacity=5000,
                 batch_size=50):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.batch_size = batch_size

        self.memory_capacity = memory_capacity
        self.memory = np.zeros(shape=[self.memory_capacity, n_states*2+3])
        self.memory_counter = 0

        self.learning_step = 0
        self.update_params_iter = 50

        self._build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, shape=[None, self.n_states], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.n_states], name='s_')
        self.a = tf.placeholder(tf.int32, shape=[None], name='a')
        self.q_target = tf.placeholder(tf.float32, shape=[None], name='q_target')

        self.eval_net = self._build_net(self.s, 'eval_net')
        self.target_net = self._build_net(self.s_, 'target_net')

        with tf.variable_scope('loss'):
            action_reward = tf.reduce_sum(tf.multiply(self.eval_net, tf.one_hot(indices=self.a, depth=self.n_actions)),
                                          axis=1)
            td_error = tf.squared_difference(action_reward, self.q_target)
            self.loss_op = tf.reduce_mean(td_error)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

    def _build_net(self, inputs, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.2), tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            hidden_layer = tf.layers.dense(inputs=inputs, units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=k_init,
                                           bias_initializer=b_init,
                                           name='hidden_layer')
            state_net = tf.layers.dense(inputs=hidden_layer, units=1,
                                        activation=None,
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init,
                                        name='state_net')
            action_net = tf.layers.dense(inputs=hidden_layer, units=self.n_actions,
                                         activation=None,
                                         kernel_initializer=k_init,
                                         bias_initializer=b_init,
                                         name='action_net')
            net = state_net + (action_net - tf.reduce_mean(action_net, axis=1, keep_dims=True))
            return net

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(range(self.n_actions))
        else:
            s = s[np.newaxis, :]
            actions = self.sess.run(self.eval_net, feed_dict={self.s: s})
            action = np.argmax(actions)
        return action

    def store_transition(self, s, a, r, s_, terminal):
        transition = np.hstack((s, a, r, s_, terminal))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _update_params(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        update_params_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(update_params_op)

    def _sample_batch_memory(self):
        assert self.memory_counter >= self.batch_size, 'No Enough Memory Yet'
        if self.memory_counter <= self.memory_capacity:
            index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, 0:self.n_states]
        ba = batch_memory[:, self.n_states]
        br = batch_memory[:, self.n_states+1]
        bs_ = batch_memory[:, -(self.n_states+1):-1]
        bt = batch_memory[:, -1]
        return bs, ba, br, bs_, bt

    def learn(self):
        if self.learning_step % self.update_params_iter == 0:
            self._update_params()

        bs, ba, br, bs_, bt = self._sample_batch_memory()

        q_next_eval = self.sess.run(self.eval_net, feed_dict={self.s: bs_})
        a4next = np.argmax(q_next_eval, axis=1)

        q_next_target = self.sess.run(self.target_net, feed_dict={self.s_: bs_})
        q_next = q_next_target[np.arange(self.batch_size), a4next]
        q_target = br + self.gamma*q_next*bt

        feed_dict = {self.s: bs, self.a: ba, self.q_target: q_target}

        _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)

        self.learning_step += 1
        self.epsilon -= (self.epsilon - self.epsilon_min)/self.epsilon_decay_step



