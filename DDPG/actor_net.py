#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class Actor(object):
    def __init__(self, n_states, n_actions, action_low_bound, action_high_bound,
                 learning_rate=0.001, tau=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_low_bound = action_low_bound
        self.action_high_bound = action_high_bound
        self.learning_rate = learning_rate
        self.tau = tau

        graph = tf.Graph()
        with graph.as_default():
            self._build_graph()

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.critic_grads = tf.placeholder(tf.float32, [None, self.n_actions], name='grads_ca')
        self.low_action = tf.constant(self.action_low_bound, dtype=tf.float32)
        self.high_action = tf.constant(self.action_high_bound, dtype=tf.float32)

        self.ae_net, self.at_net = self._build_net('eval'), self._build_net('target')

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        self.hard_params_update_op = [tf.assign(at, ae) for at, ae in zip(self.at_params, self.ae_params)]
        self.soft_params_update_op = [tf.assign(at, self.tau*ae+(1-self.tau)*at) for at, ae in zip(self.at_params, self.ae_params)]

        with tf.variable_scope('optimization_target'):
            self.actor_gradient = tf.gradients(self.ae_net, self.ae_params, -self.critic_grads)

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(zip(self.actor_gradient, self.ae_params))

        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.ae_net, var_list=self.ae_params)

    def _build_net(self, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        h1_units, h2_units = 32, 64

        with tf.variable_scope(scope):
            w1 = tf.get_variable(name='w1', shape=[self.n_states, h1_units], initializer=k_init)
            b1 = tf.get_variable(name='b1', shape=[h1_units], initializer=b_init)
            h1 = tf.nn.softplus(tf.matmul(self.s, w1) + b1)

            w2 = tf.get_variable(name='w2', shape=[h1_units, h2_units], initializer=k_init)
            b2 = tf.get_variable(name='b2', shape=[h2_units], initializer=b_init)
            h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)

            w3 = tf.get_variable(name='w3', shape=[h2_units, self.n_actions], initializer=k_init)
            b3 = tf.get_variable(name='b3', shape=[self.n_actions], initializer=b_init)
            output_layer = tf.matmul(h2, w3) + b3
            actor_net = tf.clip_by_value(output_layer, self.low_action, self.high_action)
        return actor_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.session.run(self.ae_net, feed_dict={self.s: s})[0]
        return action

    def choose_actions(self, s):
        actions = self.session.run(self.at_net, feed_dict={self.s: s})
        return actions

    def learn(self, s, critic_grads):
        self.session.run(self.soft_params_update_op)
        self.session.run(self.train_op, feed_dict={self.s: s, self.critic_grads: critic_grads})
        # print ('actor graph ops', len(self.session.graph.get_operations()))

