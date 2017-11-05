#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt

GAME = 'CartPole-v0'
EPOCHS = 300

env = gym.make(GAME)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n


def run():
    plt.ion()
    total_r = 0
    total_step = 0
    ep_r_hist = []
    for epoch in range(EPOCHS):
        env.reset()
        while True:
            a = np.random.choice(range(n_actions))
            s_, r, done, info = env.step(a)
            r = -10 if done else r

            total_r += r
            total_step += 1

            if done:
                break

        avg_ep_r = total_r / (epoch + 1)
        ep_r_hist.append(avg_ep_r)
        plt.cla()
        plt.plot(ep_r_hist)
        plt.pause(0.00001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
