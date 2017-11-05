#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sum_tree import SumTree


class Memory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.tree = SumTree(capacity=capacity)

        self.alpha = 0.6
        self.beta = 0.4
        self.p_epsilon = 1e-4
        self.batch_size = 50

    def _get_priority(self, priorities):
        priorities += self.p_epsilon
        priorities = np.minimum(priorities, 1.0)
        priorities = np.power(priorities, self.alpha)
        return priorities

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(transition, max_p)

    def sample(self):
        avg_p = self.tree.total_p()/self.batch_size
        batch_tree_idx, batch_p, batch_data = [], [], []
        for i in range(self.batch_size):
            a, b = avg_p*i, avg_p*(i+1)
            s = np.random.uniform(a, b)
            tree_idx, p, data = self.tree.sample(s)
            batch_tree_idx.append(tree_idx)
            batch_p.append(p)
            batch_data.append(data)
        batch_p /= self.tree.total_p()
        batch_weight = np.power(batch_p*self.capacity, -self.beta)
        batch_weight = batch_weight/max(batch_weight)
        batch_tree_idx, batch_data, batch_weight = map(np.array, [batch_tree_idx, batch_data, batch_weight])
        return batch_tree_idx, batch_data, batch_weight

    def update(self, tree_idx, priorities):
        priorities = self._get_priority(priorities)
        for index, p in zip(tree_idx, priorities):
            self.tree.update(index, p)


class DDQN(object):
    def __init__(self, n_states, n_actions, priority=True, gamma=0.95, learning_rate=0.0005,
                 epsilon=1, epsilon_min=0.01, exploration_step=300, batch_size=50,
                 capacity=10000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.priority = priority

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.exploration_step = exploration_step
        self.batch_size = batch_size

        self.learning_step = 0
        self.params_replace_iter = 50

        self.memory_capacity = capacity
        if self.priority:
            self.memory = Memory(capacity=self.memory_capacity, batch_size=batch_size)
        else:
            self.memory = np.zeros(shape=[capacity, self.n_states*2+3])
            self.memory_counter = 0

        self._build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.s = tf.placeholder(tf.float32, shape=[None, self.n_states], name='s')
        self.a = tf.placeholder(tf.int32, shape=[None], name='a')
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.n_states], name='s')
        self.q_target = tf.placeholder(tf.float32, shape=[None], name='td_error')
        self.is_weight = tf.placeholder(tf.float32, shape=[None], name='importance_sampling_weights')

        self.eval_net = self._build_net(inputs=self.s, scope='eval_net')
        self.target_net = self._build_net(inputs=self.s_, scope='target_net')

        with tf.variable_scope('loss'):
            action_reward = tf.reduce_sum(tf.multiply(self.eval_net, tf.one_hot(indices=self.a, depth=self.n_actions)),
                                          axis=1)
            self.td_error = tf.squared_difference(action_reward, self.q_target)
            if self.priority:
                self.loss_op = tf.reduce_mean(self.is_weight * self.td_error)
            else:
                self.loss_op = tf.reduce_mean(self.td_error)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

    def _build_net(self, inputs, scope):
        k_init, b_init = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            hl_1 = tf.layers.dense(inputs=inputs, units=64,
                                   activation=tf.nn.relu,
                                   kernel_initializer=k_init,
                                   bias_initializer=b_init,
                                   name='hl_1'
                                   )
            net = tf.layers.dense(inputs=hl_1, units=self.n_actions,
                                  activation=None,
                                  kernel_initializer=k_init,
                                  bias_initializer=b_init,
                                  name='net')
            return net

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(range(self.n_actions))
        else:
            actions = self._predict_q(s)
            action = np.argmax(actions)
        return action

    def _predict_q(self, s):
        s = s[np.newaxis, :]
        q = self.sess.run(self.eval_net, feed_dict={self.s: s})
        return q

    def store_transition(self, s, a, r, s_, terminal):
        transition = np.hstack((s, a, r, s_, terminal))
        if self.priority:
            self.memory.store(transition)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
            self.memory_counter += 1

    def _update_params(self):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        update_params_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(update_params_op)

    def learn(self):
        if self.learning_step % self.params_replace_iter == 0:
            self._update_params()

        if self.priority:
            tree_idx, batch_memory, is_weights = self.memory.sample()
        else:
            if self.memory_counter <= self.memory_capacity:
                index = np.random.choice(self.memory_counter, self.batch_size)
            else:
                index = np.random.choice(self.memory_capacity, self.batch_size)
            batch_memory = self.memory[index, :]

        bs = batch_memory[:, :self.n_states]
        ba = batch_memory[:, self.n_states]
        br = batch_memory[:, self.n_states+1]
        bs_ = batch_memory[:, -(self.n_states+1):-1]
        bt = batch_memory[:, -1]

        q_next_eval = self.sess.run(self.eval_net, feed_dict={self.s: bs_})
        a4next = np.argmax(q_next_eval, axis=1)

        q_next_target = self.sess.run(self.target_net, feed_dict={self.s_: bs_})
        q_next_target = q_next_target[np.arange(self.batch_size), a4next]

        q_target = br + self.gamma*q_next_target*bt

        if self.priority:
            fetches = [self.train_op, self.td_error, self.loss_op]
            feed_dict = {self.s: bs, self.a: ba, self.q_target: q_target, self.is_weight: is_weights}
            _, td_error, loss = self.sess.run(fetches=fetches, feed_dict=feed_dict)

            self.memory.update(tree_idx=tree_idx, priorities=td_error)
        else:
            fetches = [self.train_op, self.loss_op]
            feed_dict = {self.s: bs, self.a: ba, self.q_target: q_target}
            _, loss = self.sess.run(fetches=fetches, feed_dict=feed_dict)

        self.learning_step += 1
        self.epsilon -= (self.epsilon - self.epsilon_min)/self.exploration_step




