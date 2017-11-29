#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 18:22
# @Author  : Wang Shuailong

import numpy as np
from brain import Brain


class Agent(object):
    def __init__(self, number, sess, env, gamma=0.9, max_epochs=1000, max_ep_steps=100, params_update_iter=10):
        self.number = number
        self.name = 'agent_' + str(number)
        self.env = env
        self.agent = Brain(self.name, sess, env)  # create ACNet for each worker
        self.sess = sess
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.max_ep_steps = max_ep_steps
        self.params_update_iter = params_update_iter

    def _discounted_reward(self, v_, r):
        buffer_r = np.zeros_like(r)
        for i in reversed(range(len(r))):
            v_ = r[i] + v_ * self.gamma
            buffer_r[i] = v_
        return buffer_r

    def choose_action(self, s):
        action = self.agent.choose_action(s)
        return action

    def predict_value(self, s):
        v = self.agent.predict_value(s)
        return v

    def learn(self, buffer_s, buffer_a, buffer_v_target):
        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)

        self.agent.update_global_params(buffer_s, buffer_a, buffer_v_target)  # actual training step, update global ACNet

    def work(self):
        buffer_s, buffer_a, buffer_r = [], [], []
        total_r = 0

        for epoch in range(self.max_epochs):
            s = self.env.reset()
            ep_step = 0
            while True:
                a = self.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_step == self.max_ep_steps - 1 else False

                total_r += r

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)

                if ep_step % self.params_update_iter == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.predict_value(s_)

                    discounted_r = self._discounted_reward(v_s_, buffer_r)

                    self.learn(buffer_s, buffer_a, discounted_r)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.agent.update_local_params()

                s = s_
                ep_step += 1
                if done:
                    break

            if self.number == 0:
                avg_ep_r = total_r/(epoch+1)
                print('Episode %s Avg Reward %s' % (epoch, avg_ep_r))

