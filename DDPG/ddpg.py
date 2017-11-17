#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from actor_net import Actor
from critic_net import Critic
from memory import Memory


class DDPG(object):
    def __init__(self, n_states, n_actions, action_low_bound, action_high_bound, memory_size, batch_size=50):
        self.actor = Actor(n_states, n_actions, action_low_bound, action_high_bound)
        self.critic = Critic(n_states, n_actions)
        self.memory = Memory(n_states, memory_size, batch_size)
        self.batch_size = batch_size
        self.memory_counter = self.memory.memory_counter
        self.global_step = 0

    def train(self, s, a, r, s_):
        a_ = self.actor.choose_actions(s_)
        q_target = self.critic.predict_q(s_, a_)
        critic_loss, critic_gradient = self.critic.learn(s, a, r, q_target)
        self.actor.learn(s, critic_gradient)
        self.global_step += 1

    def choose_action(self, s):
        action = self.actor.choose_action(s)
        return action

    def choose_actions(self, s):
        actions = self.actor.choose_actions(s)
        return actions

    def store_memory(self, s, a, r, s_):
        self.memory.store_memory(s, a, r, s_)
        self._update_memory_counter()

    def sample_memory(self):
        bs, ba, br, bs_ = self.memory.sample_memory()
        return bs, ba, br, bs_

    def _update_memory_counter(self):
        self.memory_counter = self.memory.memory_counter

