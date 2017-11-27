#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import gym


class DDPG(object):
    def __init__(self, n_states, n_actions, action_low_bound, action_high_bound, gamma=0.99,
                 actor_lr=0.002, critic_lr=0.002, tau=0.01, memory_size=10000, batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_low_bound = action_low_bound
        self.action_high_bound = action_high_bound
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau

        self.memory_size = memory_size
        self.memory = np.zeros([self.memory_size, self.n_states*2 + self.n_actions + 1])
        self.memory_counter = 0
        self.batch_size = batch_size

        self.action_noise = 3
        self.action_noise_decay = 0.9995
        self.learning_counter = 0

        self._build_graph()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name='s_')
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')

        self.low_action = tf.constant(self.action_low_bound, dtype=tf.float32)
        self.high_action = tf.constant(self.action_high_bound, dtype=tf.float32)

        self.actor_net = self._build_actor_net(s=self.s, trainable=True, scope='actor_eval')
        self.actor_target_net = self._build_actor_net(s=self.s_, trainable=False, scope='actor_target')

        self.critic_net = self._build_critic_net(s=self.s, a=self.actor_net, trainable=True, scope='critic_eval')
        self.critic_target_net = self._build_critic_net(s=self.s_, a=self.actor_target_net, trainable=False,
                                                        scope='critic_target')

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea),
                              tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        with tf.variable_scope('critic_loss'):
            q_target = self.r + self.gamma*self.critic_target_net
            self.critic_loss_op = tf.reduce_mean(tf.squared_difference(q_target, self.critic_net))
        with tf.variable_scope('critic_train'):
            self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss_op,
                                                                                   var_list=self.ce_params)
        with tf.variable_scope('actor_loss'):
            self.actor_loss_op = -tf.reduce_mean(self.critic_net)  # maximize q

        with tf.variable_scope('actor_train'):
            self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss_op,
                                                                                 var_list=self.ae_params)

    def _build_actor_net(self, s, trainable, scope):

        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        h1_units = 64

        with tf.variable_scope(scope):
            w1 = tf.get_variable(name='w1', shape=[self.n_states, h1_units], initializer=k_init, trainable=trainable)
            b1 = tf.get_variable(name='b1', shape=[h1_units], initializer=b_init, trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            w2 = tf.get_variable(name='w2', shape=[h1_units, self.n_actions], initializer=k_init, trainable=trainable)
            b2 = tf.get_variable(name='b2', shape=[self.n_actions], initializer=b_init, trainable=trainable)
            actor_net = tf.matmul(h1, w2) + b2
            actor_net = tf.clip_by_value(actor_net, self.low_action, self.high_action)
        return actor_net

    def _build_critic_net(self, s, a, trainable, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.2), tf.constant_initializer(0.1)
        h1_units = 64
        with tf.variable_scope(scope):
            w1s = tf.get_variable(name='w1s', shape=[self.n_states, h1_units], initializer=k_init, trainable=trainable)
            w1a = tf.get_variable(name='w1a', shape=[self.n_actions, h1_units], initializer=k_init, trainable=trainable)
            b1 = tf.get_variable(name='b1_e', shape=[h1_units], initializer=b_init, trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(s, w1s) + tf.matmul(a, w1a) + b1)

            w2 = tf.get_variable(name='w2', shape=[h1_units, 1], initializer=k_init, trainable=trainable)
            b2 = tf.get_variable(name='b2', shape=[1], initializer=b_init, trainable=trainable)
            critic_net = tf.matmul(h1, w2) + b2
        return critic_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action_probs = self.session.run(self.actor_net, feed_dict={self.s: s})
        action = action_probs[0]
        action = np.clip(np.random.normal(action, self.action_noise), -2, 2)
        self.action_noise *= self.action_noise_decay
        return action

    def learn(self):
        self.session.run(self.soft_replace)

        bs, ba, br, bs_ = self.sample_memory()

        fetches = [self.actor_train_op, self.critic_train_op]

        self.session.run(fetches=fetches, feed_dict={self.s: bs, self.actor_net: ba,
                                                     self.r: br, self.s_: bs_})
        self.learning_counter += 1

    def store_memory(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_memory(self):
        assert self.memory_counter >= self.batch_size
        if self.memory_counter <= self.memory_size:
            index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, :self.n_states]
        ba = batch_memory[:, self.n_states: self.n_states+self.n_actions]
        br = batch_memory[:, self.n_states+self.n_actions]
        bs_ = batch_memory[:, -self.n_states:]

        br = br[:, np.newaxis]

        return bs, ba, br, bs_
