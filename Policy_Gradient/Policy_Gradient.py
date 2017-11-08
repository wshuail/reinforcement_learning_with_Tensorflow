#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class PolicyGradient(object):
    def __init__(self, n_states, n_actions, learning_rate=0.01, gamma=0.95):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.ep_s, self.ep_a, self.ep_r = [], [], []

        self._build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.learning_step = 0

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.r = tf.placeholder(tf.float32, [None], name='r')

        k_init, b_init = tf.random_uniform_initializer(0., 0.2), tf.constant_initializer(0.1)

        hidden_layer = tf.layers.dense(inputs=self.s, units=64,
                                       activation=tf.nn.tanh,
                                       kernel_initializer=k_init,
                                       bias_initializer=b_init,
                                       name='hidden_layer')
        action_probs = tf.layers.dense(inputs=hidden_layer, units=self.n_actions,
                                       activation=None,
                                       kernel_initializer=k_init,
                                       bias_initializer=b_init,
                                       name='action_probs')
        self.net = tf.nn.softmax(action_probs, name='net')

        with tf.variable_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.net)*tf.one_hot(self.a, self.n_actions), axis=1)
            self.loss_op = tf.reduce_mean(self.r * neg_log_prob)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action_probs = self.session.run(self.net, feed_dict={self.s: s})
        action = np.random.choice(range(self.n_actions), p=action_probs.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)

    def learn(self):
        dn_ep_r = self._discount_norm_reward()
        _, cost = self.session.run([self.train_op, self.loss_op],
                                   feed_dict={self.s: np.vstack(self.ep_s),
                                              self.a: np.array(self.ep_a),
                                              self.r: np.array(dn_ep_r)})
        self.ep_s, self.ep_a, self.ep_r = [], [], []

        self.learning_step += 1

    def _discount_norm_reward(self):
        dn_ep_r = np.zeros_like(self.ep_r)
        running_add = 0
        for i in reversed(range(len(self.ep_r))):
            running_add = running_add * self.gamma + self.ep_r[i]
            dn_ep_r[i] = running_add
        dn_ep_r -= np.mean(dn_ep_r)
        dn_ep_r /= np.std(dn_ep_r)
        return dn_ep_r
