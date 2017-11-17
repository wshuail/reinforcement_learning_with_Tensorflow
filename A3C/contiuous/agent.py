#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from brain import Brain


class Agent(object):
    def __init__(self, number, env, coordinator, epsilon=1, epsilon_min=0.01,
                 epsilon_decay_step=300, gamma=0.99, params_update_iter=30):
        self.name = 'agent_' + str(number)
        self.number = number
        self.env = env
        self.coordinator = coordinator
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.a_low_bound = env.a_low_bound
        self.a_high_bound = env.a_high_bound
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.gamma = gamma
        self.params_update_iter = params_update_iter
        self.agent_episode = 0

        self.agent = Brain(scope=self.name, env=env)
        self.global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        self.local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

    def choose_action(self, s):
        if np.random.uniform() <= self.epsilon:
            action = np.random.uniform(self.a_low_bound, self.a_high_bound)
        else:
            action = self.agent.predict_action(s)
        return action

    def _discounted_reward(self, v_, r):
        buffer_r = np.zeros_like(r)
        for i in reversed(range(len(r))):
            v_ = r[i] + v_ * self.gamma
            buffer_r[i] = v_
        return buffer_r

    def learn(self, s, a, dr):
        s, a, dr = np.vstack(s), np.vstack(a), np.vstack(dr)
        ops = [self.agent.policy_train_op, self.agent.value_loss_op,
               self.agent.policy_loss_op, self.agent.critic_net, self.agent.mu, self.agent.sigma]
        feed_dict = {self.agent.s: s, self.agent.a: a, self.agent.q_target: dr}
        _, value_loss, policy_loss, v, mu, sigma = self.agent.session.run(ops, feed_dict=feed_dict)

        if self.epsilon >= self.epsilon_min:
            self.epsilon -= (self.epsilon-self.epsilon_min)/self.epsilon_decay_step
        return value_loss, policy_loss

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

        total_r, total_value_loss, total_policy_loss = 0, 0, 0

        while not self.coordinator.should_stop():

            self._update_local_op()

            s = self.env.reset()
            ep_r = 0
            ep_step = 0

            while True:
                a = self.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_step >= 100 else False

                ep_r += r
                total_r += r
                ep_step += 1
                total_step += 1

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if ep_step % self.params_update_iter == 0 or done:
                    if done:
                        v_ = 0
                    else:
                        v_ = self.agent.predict_value(s_)
                    discounted_r = self._discounted_reward(v_=v_, r=buffer_r)
                    value_loss, policy_loss = self.learn(s=buffer_s, a=buffer_a, dr=discounted_r)

                    total_value_loss += value_loss
                    total_policy_loss += policy_loss

                    buffer_s, buffer_a, buffer_r = [], [], []

                    self._update_global_op()

                s = s_

                if done or ep_step >= 100:
                    self.agent_episode += 1
                    break

            if self.name == 'agent_0':
                avg_ep_r = int(total_r/(self.agent_episode + 1))
                avg_ep_value_loss = int(total_value_loss / (self.agent_episode + 1))
                avg_ep_policy_loss = int(total_policy_loss / (self.agent_episode + 1))
                print('Episode %d Avg Reward %s Value Loss %s Policy Loss %s' %
                      (self.agent_episode, avg_ep_r, avg_ep_value_loss, avg_ep_policy_loss))


