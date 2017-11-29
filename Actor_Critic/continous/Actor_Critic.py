#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, n_states, n_actions, low_action_bound, high_action_bound, learning_rate=0.0001,
                 epsilon=1, epsilon_min=0.01, epsilon_decay_step=200):
        self.n_states = n_states
        self.n_actions = n_actions
        self.low_action_bound = low_action_bound
        self.high_action_bound = high_action_bound
        self.learning_rate = learning_rate

        self.learning_step = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step

        self._build_graph()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.float32, None, name='a')
        self.td = tf.placeholder(tf.float32, None, name='td')

        mu, sigma = self._build_net()

        with tf.variable_scope('actor_net'):
            mu, sigma = tf.squeeze(mu*2), tf.squeeze(sigma+1e-4)
            normal_dist = tf.distributions.Normal(mu, sigma)
            action = normal_dist.sample(1)
            self.actor_net = tf.clip_by_value(action, self.low_action_bound, self.high_action_bound)

        with tf.variable_scope('loss'):
            self.neg_log_prob = normal_dist.log_prob(self.a)
            self.policy_loss = self.neg_log_prob*self.td
            self.loss_op = -tf.reduce_mean(self.policy_loss + tf.stop_gradient(0.01*normal_dist.entropy()))

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

    def _build_net(self):
        with tf.variable_scope('actor_net'):
            k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
            hl_1 = tf.layers.dense(inputs=self.s, units=64,
                                   activation=tf.nn.relu,
                                   kernel_initializer=k_init,
                                   bias_initializer=b_init,
                                   name='hl_1'
                                   )
            hl_2 = tf.layers.dense(inputs=hl_1, units=64,
                                   activation=tf.nn.relu,
                                   kernel_initializer=k_init,
                                   bias_initializer=b_init,
                                   name='hl_2'
                                   )
            mu = tf.layers.dense(inputs=hl_2,
                                 units=1,
                                 activation=tf.nn.tanh,
                                 kernel_initializer=k_init,
                                 bias_initializer=b_init,
                                 name='mu')
            sigma = tf.layers.dense(inputs=hl_2,
                                    units=1,
                                    activation=tf.nn.softplus,
                                    kernel_initializer=k_init,
                                    bias_initializer=b_init,
                                    name='sigma')
        return mu, sigma

    def choose_action(self, s):
        if np.random.uniform() > self.epsilon:
            s = s[np.newaxis, :]
            action = self.session.run(self.actor_net, feed_dict={self.s: s})
        else:
            action = np.random.uniform(self.low_action_bound, self.high_action_bound, 1)
        return action

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td: td}
        self.session.run(fetches=self.train_op, feed_dict=feed_dict)

        self.learning_step += 1
        self.epsilon -= (self.epsilon - self.epsilon_min)/self.learning_step


class Critic(object):
    def __init__(self, n_states, gamma=0.95, learning_rate=0.0001):
        self.n_states = n_states
        self.gamma = gamma
        self.learning_rate = learning_rate

        self._build_graph()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.r = tf.placeholder(tf.float32, None, name='r')
        self.v_ = tf.placeholder(tf.float32, None, name='v_')

        self.critic_net = self._build_net()

        with tf.name_scope('loss'):
            self.td_error = self.r + self.gamma*self.v_ - self.critic_net
            self.loss_op = tf.reduce_mean(tf.square(self.td_error))

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

    def _build_net(self):
        with tf.variable_scope('critic_net'):
            k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
            hl_1 = tf.layers.dense(inputs=self.s, units=64,
                                   activation=tf.nn.relu,
                                   kernel_initializer=k_init,
                                   bias_initializer=b_init,
                                   name='hl_1'
                                   )
            hl_2 = tf.layers.dense(inputs=hl_1, units=64,
                                   activation=tf.nn.relu,
                                   kernel_initializer=k_init,
                                   bias_initializer=b_init,
                                   name='hl_2'
                                   )
            critic_net = tf.layers.dense(inputs=hl_2,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=k_init,
                                         bias_initializer=b_init,
                                         name='critic_net')
        return critic_net

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.session.run(self.critic_net, feed_dict={self.s: s_})
        feed_dict = {self.s: s, self.r: r, self.v_: v_}
        _, td_error = self.session.run(fetches=[self.train_op, self.td_error], feed_dict=feed_dict)
        return td_error
