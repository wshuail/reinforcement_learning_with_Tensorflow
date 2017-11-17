#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym


class Environment(object):
    def __init__(self, name, seed=1):
        self.name = name
        env = gym.make(name)
        env.seed(seed)
        self.env = env.unwrapped
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

