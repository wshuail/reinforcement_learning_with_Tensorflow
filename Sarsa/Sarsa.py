#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Sarsa(object):
    def __init__(self, n_states, n_actions, epsilon=1, epsilon_min=0.01,
                 epsilon_decay_step=2000, gamma=0.99, learning_rate=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.q_table = np.zeros(shape=[n_states, n_actions])

    def choose_action(self, s):
        if np.random.randn() < self.epsilon:
            action = np.random.choice(range(self.n_actions))
        else:
            action_values = self.q_table[s, ]
            action = np.argmax(action_values)
        return action

    def learn(self, s, a, r, s_, done, a_):
        q = self.q_table[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma*self.q_table[s_, a_]
        self.q_table[s, a] += self.learning_rate*(q_target - q)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon - self.epsilon_min)/self.epsilon_decay_step

    def plot_result(self, data, x_label, y_label):
        plt.plot(np.arange(len(data)), data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
