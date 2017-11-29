#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PPO(object):
    def __init__(self, n_states, n_actions, low_action_bound, high_action_bound, mode=None, actor_lr=0.0001,
                 critic_lr=0.0002, params_update_iter=32, gamma=0.9, action_scale=1, kl_lambda=0.5, kl_target=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.low_action_bound = low_action_bound
        self.high_action_bound = high_action_bound
        if mode is None:
            self.mode = 'CLIP'
        else:
            assert mode in ('CLIP', 'KL'), 'Mode should be one in CLIP or KL'
            self.mode = mode
            if self.mode == 'KL':
                assert kl_lambda is not None and kl_target is not None, 'set kl_lambda and kl_target for KL mode'
                self.kl_lambda = kl_lambda
                self.kl_target = kl_target
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.params_update_iter = params_update_iter
        self.gamma = gamma
        self.action_scale = action_scale

        self.actor_update_steps = 10
        self.critic_update_steps = 10

        self.learning_step = 0

        self.action_noise = 3
        self.action_noise_decay = 0.9995

        self._build_graph()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], name='a')
        self.dr = tf.placeholder(tf.float32, [None, 1], name='discounted_r')
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')

        with tf.variable_scope('critic_net'):
            self.critic_net = self._build_critic_net(scope='critic')
            self.td = self.dr - self.critic_net
            self.critic_loss_op = tf.reduce_mean(tf.square(self.td))
            self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss_op)

        normal_dist, new_params = self._build_actor_net(scope='new_actor', trainable=True)
        normal_dist_old, old_params = self._build_actor_net(scope='old_actor', trainable=False)

        with tf.variable_scope('sample_action'):
            self.actor_net = tf.squeeze(normal_dist.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_actor_net'):
            self.update_params_op = [old_p.assign(new_p) for new_p, old_p in zip(new_params, old_params)]

        with tf.variable_scope('actor_loss'):
            ratio = normal_dist.prob(self.a)/(normal_dist_old.prob(self.a)+1e-10)
            surrogate = ratio*self.advantage
            if self.mode == 'CLIP':
                clipping_adv = tf.clip_by_value(ratio, 1-0.2, 1+0.2)*self.advantage
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate, clipping_adv))
            else:
                kl = tf.stop_gradient(tf.contrib.distributions.kl_divergence(normal_dist_old, normal_dist))
                self.kl_mean = tf.reduce_mean(kl)
                actor_loss = -tf.reduce_mean(surrogate - self.kl_lambda*kl)

        with tf.variable_scope('actor_train'):
            self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(actor_loss)

    def _build_critic_net(self, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        units = 64
        with tf.variable_scope(scope):
            w1 = tf.get_variable(name='w1', shape=[self.n_states, units], initializer=k_init)
            b1 = tf.get_variable(name='b1', shape=[units], initializer=b_init)
            h1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            w2 = tf.get_variable(name='w2', shape=[units, 1], initializer=k_init)
            b2 = tf.get_variable(name='b2', shape=[1], initializer=b_init)
            critic_net = tf.matmul(h1, w2) + b2

        return critic_net

    def _build_actor_net(self, scope, trainable):
        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        units = 128
        with tf.variable_scope(scope):
            w1 = tf.get_variable(name='w1', shape=[self.n_states, units], initializer=k_init, trainable=trainable)
            b1 = tf.get_variable(name='b1', shape=[units], initializer=b_init, trainable=trainable)
            h1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            w2_mu = tf.get_variable(name='w2_mu', shape=[units, 1], initializer=k_init, trainable=trainable)
            b2_mu = tf.get_variable(name='b2_mu', shape=[1], initializer=b_init, trainable=trainable)
            mu = tf.matmul(h1, w2_mu) + b2_mu

            w2_sigma = tf.get_variable(name='w2_sigma', shape=[units, 1], initializer=k_init, trainable=trainable)
            b2_sigma = tf.get_variable(name='b2_sigma', shape=[1], initializer=b_init, trainable=trainable)
            sigma = tf.matmul(h1, w2_sigma) + b2_sigma

            mu, sigma = self.action_scale*tf.nn.tanh(mu), tf.nn.softplus(sigma)
            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return norm_dist, params

    def learn(self, s, a, r):
        self.session.run(self.update_params_op)
        advantage = self.session.run(self.td, feed_dict={self.s: s, self.dr: r})

        # update actor
        actor_feed_dict = {self.s: s, self.a: a, self.advantage: advantage}
        if self.mode == 'CLIP':
            [self.session.run(self.actor_train_op, feed_dict=actor_feed_dict) for _ in range(self.actor_update_steps)]
        else:
            kl_mean_value = 0
            for _ in range(self.actor_update_steps):
                _, kl_mean_value = self.session.run([self.actor_train_op, self.kl_mean],
                                                    feed_dict=actor_feed_dict)
                if kl_mean_value > 4*self.kl_target:
                    break
            if kl_mean_value < self.kl_target/1.5:
                self.kl_lambda /= 2
            elif kl_mean_value >= self.kl_target/1.5:
                self.kl_lambda *= 2
            self.kl_lambda = np.clip(self.kl_lambda, 1e-4, 10)

        # update critic
        [self.session.run(self.critic_train_op, {self.s: s, self.dr: r}) for _ in range(self.critic_update_steps)]

    def predict_action(self, s):
        s = s[np.newaxis, :]
        action = self.session.run(self.actor_net, feed_dict={self.s: s})[0]
        # action = np.clip(np.random.normal(action, self.action_noise), -2, 2)
        # self.action_noise *= self.action_noise_decay
        action = np.clip(action, self.low_action_bound, self.high_action_bound)
        return action

    def predict_value(self, s):
        s = s[np.newaxis, :]
        v = self.session.run(self.critic_net, feed_dict={self.s: s})[0, 0]
        return v

    def discounted_reward(self, v_, r):
        discounted_r = np.zeros_like(r)
        for i in reversed(range(len(r))):
            v_ = r[i] + v_ * self.gamma
            discounted_r[i] = v_
        return discounted_r


