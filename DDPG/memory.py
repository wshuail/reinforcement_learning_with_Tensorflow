#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Memory(object):
    def __init__(self, n_states, memory_size, batch_size):
        self.n_states = n_states
        self.memory_size = memory_size
        self.memory = np.zeros([self.memory_size, self.n_states*2 + 2])
        self.memory_counter = 0
        self.batch_size = batch_size

    def store_memory(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_memory(self):
        assert self.memory_counter >= self.batch_size
        if self.memory_counter <= self.memory_size:
            index = np.random.choice(self.batch_size, self.memory_counter)
        else:
            index = np.random.choice(self.batch_size, self.memory_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, :self.n_states]
        ba = batch_memory[:, self.n_states]
        br = batch_memory[:, self.n_states+1]
        bs_ = batch_memory[:, -self.n_states:]

        ba = np.vstack(ba)

        return bs, ba, br, bs_

