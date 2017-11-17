#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Brain(object):
    def __init__(self, scope, env, gamma=0.9, critic_lr=0.005, actor_lr=0.0001,
                 epsilon=0.1, output_graph=False):
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.a_low_bound = env.a_low_bound
        self.a_high_bound = env.a_high_bound
        self.r_low_bound = env.r_low_bound
        self.r_high_bound = env.r_high_bound
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.epsilon = epsilon

        self.CRITIC_LOSS = 0.5
        self.ENTROPY_LOSS = 0.01

        self._build_graph(scope)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if output_graph:
            log_dir = './log'
            tf.summary.FileWriter(log_dir, self.session.graph)

    def _build_graph(self, scope):

        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.float32, [None, 1], name='a')
        self.q_target = tf.placeholder(tf.float32, [None, 1], name='v_')

        self.a_low_bound = tf.constant(self.a_low_bound, dtype=tf.float32)
        self.a_high_bound = tf.constant(self.a_high_bound, dtype=tf.float32)
        self.r_low_bound = tf.constant(self.r_low_bound, dtype=tf.float32)
        self.r_high_bound = tf.constant(self.r_high_bound, dtype=tf.float32)

        if scope == 'global':
            _, _, _ = self._build_net(scope)
        else:
            self.mu, self.sigma, self.critic_net = self._build_net(scope)

            with tf.variable_scope('actor_net'):
                self.mu, self.sigma = tf.squeeze(self.mu * 2), tf.squeeze(self.sigma + 1e-20)
                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
                action = normal_dist.sample(1)
                self.actor_net = tf.clip_by_value(action, self.a_low_bound, self.a_high_bound)
            with tf.variable_scope('critic_net'):
                self.critic_net = tf.clip_by_value(self.critic_net, self.r_low_bound, self.r_high_bound)

            with tf.variable_scope('value_loss'):
                self.critic_net = tf.squeeze(self.critic_net)
                self.td_error = tf.subtract(self.critic_net, self.q_target)
                self.value_loss_op = tf.reduce_mean(tf.square(self.td_error))
            # with tf.variable_scope('value_train'):
            #     self.value_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.value_loss_op)

            with tf.variable_scope('policy_loss'):
                self.entropy_op = -tf.reduce_sum(0.5 * (tf.log(2.*np.pi*self.sigma) + 1.))*self.ENTROPY_LOSS

                # Gaussian negative-log-likelihood
                bs = tf.to_float(tf.size(self.a) / self.n_actions)  # batch_size
                x_power = tf.square(tf.subtract(self.actor_net, self.mu))*tf.exp(-tf.log(self.sigma))*(-0.5)
                GausNLL = (tf.reduce_sum(tf.log(self.sigma)) + bs*tf.log(2.*np.pi))/2 - tf.reduce_sum(x_power, axis=1)
                self.policy_loss = tf.reduce_mean(tf.multiply(GausNLL, tf.stop_gradient(self.td_error)))
                self.policy_loss_op = self.policy_loss + self.entropy_op

                self.loss_op = self.value_loss_op*self.CRITIC_LOSS + self.policy_loss_op

            with tf.variable_scope('train'):
                self.policy_train_op = tf.train.RMSPropOptimizer(self.critic_lr).minimize(self.loss_op)

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            k_init, b_init = tf.random_normal_initializer(0., .1), tf.constant_initializer(0.1)
            with tf.variable_scope('actor'):
                actor_hidden = tf.layers.dense(inputs=self.s, units=128,
                                               activation=tf.nn.relu6,
                                               kernel_initializer=k_init,
                                               bias_initializer=b_init,
                                               name='actor_hidden')
                mu = tf.layers.dense(inputs=actor_hidden,
                                     units=1,
                                     activation=tf.nn.tanh,
                                     kernel_initializer=k_init,
                                     bias_initializer=b_init,
                                     name='mu')
                sigma = tf.layers.dense(inputs=actor_hidden,
                                        units=1,
                                        activation=tf.nn.softplus,
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init,
                                        name='sigma')
            with tf.variable_scope('critic'):
                critic_hidden = tf.layers.dense(inputs=self.s, units=128,
                                                activation=tf.nn.relu6,
                                                kernel_initializer=k_init,
                                                bias_initializer=b_init,
                                                name='critic_hidden')
                critic_net = tf.layers.dense(inputs=critic_hidden,
                                             units=1,
                                             kernel_initializer=k_init,
                                             bias_initializer=b_init,
                                             name='critic_net')  # state value

        return mu, sigma, critic_net

    def predict_action(self, s):
        s = s[np.newaxis, :]
        mu, sigma, action = self.session.run([self.mu, self.sigma, self.actor_net], feed_dict={self.s: s})
        return action

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
