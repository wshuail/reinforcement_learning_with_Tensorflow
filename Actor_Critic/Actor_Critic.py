#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class Actor(object):
    def __init__(self, n_states, n_actions, learning_rate=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        self._build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, None, name='a')
        self.td_error = tf.placeholder(tf.float32, None, name='td_error')  # a number

        k_init, b_init = tf.random_normal_initializer(0., 0.2), tf.constant_initializer(0.1)

        with tf.variable_scope('actor_net'):
            hidden_layer = tf.layers.dense(inputs=self.s, units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=k_init,
                                           bias_initializer=b_init,
                                           name='hidden_layer')
            self.actor_net = tf.layers.dense(inputs=hidden_layer,
                                             units=self.n_actions,
                                             activation=tf.nn.softmax,
                                             kernel_initializer=k_init,
                                             bias_initializer=b_init,
                                             name='actor_net')

        with tf.variable_scope('loss'):
            action_prob = tf.reduce_sum(tf.multiply(self.actor_net, tf.one_hot(indices=self.a, depth=self.n_actions)),
                                        axis=1)
            log_prob = tf.log(action_prob)
            eligibility = log_prob*self.td_error
            self.loss_op = -tf.reduce_sum(eligibility)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions_probs = self.session.run(self.actor_net, feed_dict={self.s: s})
        action = np.random.choice(range(self.n_actions), p=actions_probs.ravel())
        return action

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]
        self.session.run(self.train_op, feed_dict={self.s: s,
                                                   self.a: a,
                                                   self.td_error: td_error})


class Critic(object):
    def __init__(self, n_states, gamma=0.97, learning_rate=0.001):
        self.n_states = n_states
        self.gamma = gamma
        self.learning_rate = learning_rate

        self._build_net()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.r = tf.placeholder(tf.float32, None, name='r')  # r is a single number
        self.v_ = tf.placeholder(tf.float32, [None, 1], name='v_')  # also [1, 1]

        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

        with tf.variable_scope('critic_net'):
            hidden_layer = tf.layers.dense(inputs=self.s, units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=k_init,
                                           bias_initializer=b_init,
                                           name='hidden_layer')
            self.critic_net = tf.layers.dense(inputs=hidden_layer,
                                              units=1,
                                              activation=None,
                                              kernel_initializer=k_init,
                                              bias_initializer=b_init,
                                              name='critic_net')
        with tf.variable_scope('loss'):
            self.td_error = self.r + self.gamma*self.v_ - self.critic_net
            self.loss_op = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.session.run(self.critic_net, feed_dict={self.s: s_})

        _, td_error = self.session.run([self.train_op, self.td_error],
                                       feed_dict={self.s: s,
                                                  self.r: r,
                                                  self.v_: v_})
        return td_error


