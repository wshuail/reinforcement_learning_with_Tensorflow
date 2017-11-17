#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class Critic(object):
    def __init__(self, n_states, n_actions, gamma=0.99, learning_rate=0.001, tau=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau

        graph = tf.Graph()
        with graph.as_default():
            self._build_graph()

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='a')
        self.r = tf.placeholder(tf.float32, [None], name='r')
        self.q_target = tf.placeholder(tf.float32, [None, 1], name='q_target')

        self.ce_net, self.ct_net = self._build_net(scope='eval'), self._build_net(scope='target')

        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        self.hard_params_update_op = [tf.assign(ct, ce) for ct, ce in zip(self.ct_params, self.ce_params)]
        self.soft_params_update_op = [tf.assign(ct, self.tau*ce+(1-self.tau)*ct) for ct, ce in zip(self.ct_params, self.ce_params)]

        with tf.variable_scope('loss'):
            td_error = self.r + self.gamma*self.q_target - self.ce_net
            self.loss_op = tf.reduce_mean(tf.square(td_error), axis=1)
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss_op)
        with tf.variable_scope('critic_grads'):
            # self.grads_qa = self.optimizer.compute_gradients(self.loss_op, var_list=self.a)
            self.critic_grads_op = tf.gradients(self.ce_net, self.a)[0]

    def _build_net(self, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.2), tf.constant_initializer(0.1)
        h1_units, h2_units = 32, 64
        with tf.variable_scope(scope):
            w1 = tf.get_variable(name='w1_s', shape=[self.n_states, h1_units], initializer=k_init)
            b1 = tf.get_variable(name='b1_e', shape=[h1_units], initializer=b_init)
            h1 = tf.nn.softplus(tf.matmul(self.s, w1) + b1)

            w2_h1 = tf.get_variable(name='w2_h1', shape=[h1_units, h2_units], initializer=k_init)
            w2_a = tf.get_variable(name='w2_a', shape=[self.n_actions, h2_units], initializer=k_init)
            b2 = tf.get_variable(name='b2', shape=[h2_units], initializer=b_init)
            h2 = tf.nn.tanh(tf.matmul(h1, w2_h1) + tf.matmul(self.a, w2_a) + b2)

            w3_h2 = tf.get_variable(name='w3_h2', shape=[h2_units, 1], initializer=k_init)
            b3 = tf.get_variable(name='b3', shape=[1], initializer=b_init)
            critic_net = tf.matmul(h2, w3_h2) + b3
        return critic_net

    def predict_q(self, s, a):
        q = self.session.run(self.ct_net, feed_dict={self.s: s, self.a: a})
        return q

    def learn(self, bs, ba, br, q_target):
        self.session.run(self.soft_params_update_op)
        _, loss, critic_grads = self.session.run([self.train_op, self.loss_op, self.critic_grads_op],
                                                 feed_dict={self.s: bs,
                                                            self.a: ba,
                                                            self.r: br,
                                                            self.q_target: q_target})
        # print ('critic graph ops', len(self.session.graph.get_operations()))
        return loss, critic_grads

