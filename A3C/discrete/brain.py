#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Brain(object):
    def __init__(self, scope, env, gamma=0.99, learning_rate=0.003,
                 epsilon=0.1, output_graph=False):
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.CRITIC_LOSS = 0.5
        self.ENTROPY_LOSS = 0.01

        graph = tf.Graph()
        with graph.as_default():
            self._build_graph(scope)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            if output_graph:
                log_dir = './log'
                tf.summary.FileWriter(log_dir, self.session.graph)

    def _build_graph(self, scope):

        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, [None, 1], name='a')
        self.q_target = tf.placeholder(tf.float32, [None, 1], name='v_')

        if scope == 'global':
            _, _ = self._build_net(scope)

        else:
            self.actor_net, self.critic_net = self._build_net(scope)
            with tf.variable_scope('loss'):
                self.td_error = self.q_target - self.critic_net
                critic_loss = tf.square(self.td_error)*self.CRITIC_LOSS
                action_values = tf.one_hot(self.a, self.n_actions, dtype=tf.float32)
                policy_log = tf.log(tf.reduce_sum(self.actor_net * action_values, axis=1, keep_dims=True) + 1e-10)
                # self.policy_loss = -policy_log * tf.stop_gradient(self.td_error)
                self.policy_loss = -policy_log * self.td_error
                self.entropy = tf.reduce_sum(self.actor_net * tf.log(self.actor_net + 1e-10),
                                             axis=1, keep_dims=True)*self.ENTROPY_LOSS

                self.loss_op = tf.reduce_mean(critic_loss + self.policy_loss + self.entropy)

            with tf.variable_scope('train'):
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                self.train_op = optimizer.minimize(self.loss_op)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            w_init = tf.random_normal_initializer(0., .1)
            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(self.s, 128, tf.nn.leaky_relu, kernel_initializer=w_init, name='la')
                actor_net = tf.layers.dense(l_a, self.n_actions, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(self.s, 128, tf.nn.leaky_relu, kernel_initializer=w_init, name='lc')
                critic_net = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                # print('name of critic net: ', critic_net)

        return actor_net, critic_net

    def predict_action(self, s):
        s = s[np.newaxis, :]
        action_probs = self.session.run(self.actor_net, feed_dict={self.s: s})
        return action_probs

    def predict_value(self, s):
        s = s[np.newaxis, :]
        q = self.session.run(self.critic_net, feed_dict={self.s: s})
        return q

    def update_params(self, from_params, to_params):
        update_params_op = [tf.assign(tp, fp) for fp, tp in zip(from_params, to_params)]
        self.session.run(update_params_op)

    def get_params(self, scope):
        get_params_op = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
        params = self.session.run(get_params_op)
        return params

