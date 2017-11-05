#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, transition, p):
        data_index = self.data_pointer % self.capacity
        self.data[data_index] = transition
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, p)
        self.data_pointer += 1
        self.data_pointer %= self.capacity

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    def sample(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    def _retrieve(self, tree_idx, s):
        left = 2 * tree_idx + 1
        # print('left idx: ', left)
        right = left + 1
        # print('right idx: ', right)

        if left >= len(self.tree):
            # print('tree_idx: ', tree_idx)
            # print('data idx: ', tree_idx - self.capacity + 1)
            return tree_idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total_p(self):
        return self.tree[0]

