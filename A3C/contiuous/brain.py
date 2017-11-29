#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 9:58
# @Author  : Wang Shuailong

import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf


# Network for the Actor Critic
class Brain(object):
    def __init__(self, scope, sess, env, action_scale=2, actor_lr=0.001, critic_lr=0.001):
        self.sess = sess
        self.env = env
        self.low_action_bound = env.a_low_bound
        self.high_action_bound = env.a_high_bound
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.action_scale = action_scale
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.entropy_beta = 0.01

        self.build_graph(scope)

    def build_graph(self, scope):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], 's')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], 'a')
        self.q_target = tf.placeholder(tf.float32, [None, 1], 'q_target')
        if scope == 'global':
            self._build_net(scope=scope)
        else:
            mu, sigma, self.critic_net = self._build_net(scope=scope)

            la_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            lc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

            ga_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global/actor')
            gc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global/critic')

            with tf.name_scope('c_loss'):
                td = tf.subtract(self.q_target, self.critic_net, name='TD_error')
                self.critic_loss_op = tf.reduce_mean(tf.square(td))

            with tf.name_scope('a_loss'):
                sigma = sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                log_prob_action_adv = normal_dist.log_prob(self.a) * td
                entropy = normal_dist.entropy()
                self.policy_loss = self.entropy_beta * entropy + log_prob_action_adv
                self.policy_loss_op = tf.reduce_mean(-self.policy_loss)

            with tf.variable_scope('train'):
                self.actor_optimizer = tf.train.RMSPropOptimizer(self.actor_lr, name='RMSPropA')
                self.critic_optimizer = tf.train.RMSPropOptimizer(self.critic_lr, name='RMSPropC')

            with tf.name_scope('choose_a'):
                self.actor_net = tf.squeeze(normal_dist.sample(1), axis=0)
                self.actor_net = tf.clip_by_value(self.actor_net, self.low_action_bound, self.high_action_bound)

            with tf.name_scope('local_grad'):
                self.actor_grads = tf.gradients(self.policy_loss_op, la_params, name='actor_grads')
                self.critic_grads = tf.gradients(self.critic_loss_op, lc_params, name='critic_grads')

            with tf.name_scope('pull'):
                self.update_la_params_op = [la.assign(ga) for la, ga in zip(la_params, ga_params)]
                self.update_lc_params_op = [lc.assign(gc) for lc, gc in zip(lc_params, gc_params)]
            with tf.name_scope('push'):
                self.update_ga_params_op = self.actor_optimizer.apply_gradients(zip(self.actor_grads, ga_params))
                self.update_gc_params_op = self.critic_optimizer.apply_gradients(zip(self.critic_grads, gc_params))

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
                                     units=self.n_actions,
                                     activation=tf.nn.tanh,
                                     kernel_initializer=k_init,
                                     bias_initializer=b_init,
                                     name='mu')
                sigma = tf.layers.dense(inputs=actor_hidden,
                                        units=self.n_actions,
                                        activation=tf.nn.softplus,
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init,
                                        name='sigma')
                mu = self.action_scale*mu
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

    def update_global_params(self, s, a, dr):
        feed_dict = {self.s: s, self.a: a, self.q_target: dr}
        self.sess.run([self.update_ga_params_op, self.update_gc_params_op], feed_dict)

    def update_local_params(self):
        self.sess.run([self.update_la_params_op, self.update_lc_params_op])

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.actor_net, {self.s: s})[0]

    def predict_value(self, s):
        s = s[np.newaxis, :]
        v = self.sess.run(self.critic_net, {self.s: s})[0, 0]
        return v

