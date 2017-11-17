#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from brain import Brain

import tensorflow as tf


class Agent(object):
    def __init__(self, number, env, coordinator, epsilon=0.1, gamma=0.95,
                 params_update_iter=30, max_episodes=1000):
        self.name = 'agent_' + str(number)
        self.number = number
        self.env = env
        self.coordinator = coordinator
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.params_update_iter = params_update_iter
        self.max_episodes = max_episodes
        self.agent_episodes = 0

        self.agent = Brain(scope=self.name, env=env)
        self.global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        self.local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(range(0, self.n_actions))
        else:
            action_probs = self.agent.predict_action(s)
            action = np.random.choice(range(self.n_actions), p=action_probs.ravel())
        return action

    def _discounted_reward(self, v_, r):
        buffer_r = np.zeros_like(r)
        for i in reversed(range(len(r))):
            v_ = r[i] + v_ * self.gamma
            buffer_r[i] = v_
        return buffer_r

    def learn(self, s, a, dr):
        s, a, dr = np.vstack(s), np.vstack(a), np.vstack(dr)
        ops = [self.agent.train_op, self.agent.loss_op, self.agent.td_error, self.agent.policy_loss, self.agent.entropy]
        feed_dict = {self.agent.s: s, self.agent.a: a, self.agent.q_target: dr}
        _, loss, td_error, policy_loss, entropy = self.agent.session.run(ops, feed_dict=feed_dict)
        return loss

    def _update_params(self):
        self.global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        self.local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    def _update_local_op(self):
        self._update_params()
        self.agent.update_params(from_params=self.global_params, to_params=self.local_params)

    def _update_global_op(self):
        self._update_params()
        self.agent.update_params(from_params=self.local_params, to_params=self.global_params)

    def work(self):

        buffer_s, buffer_a, buffer_r = [], [], []

        total_step = 0
        total_r = 0

        while self.agent_episodes <= self.max_episodes:

            self._update_local_op()

            s = self.env.reset()

            ep_r = 0
            ep_step = 0

            while True:
                a = self.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done:
                    r = -10

                ep_r += r
                total_r += r

                ep_step += 1
                total_step += 1

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % self.params_update_iter == 0 or done:
                    if done:
                        v_ = 0
                    else:
                        v_ = self.agent.predict_value(s_)
                    discounted_r = self._discounted_reward(v_=v_, r=buffer_r)
                    self.learn(s=buffer_s, a=buffer_a, dr=discounted_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    self._update_global_op()

                s = s_

                if done:
                    self.agent_episodes += 1
                    break

            if self.name == 'agent_0':
                print('Agent %s Episode %d Avg Reward %s' %
                      (self.name, self.agent_episodes, int(total_r/(self.agent_episodes + 1))))




