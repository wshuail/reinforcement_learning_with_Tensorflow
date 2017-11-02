#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from Sarsa import Sarsa

EPOCHS = 5000

GAME = 'FrozenLake-v0'
env = gym.make(GAME)
env = env.unwrapped
n_states = env.observation_space.n
n_actions = env.action_space.n

agent = Sarsa(n_states=n_states, n_actions=n_actions)


def run():
    total_r = 0
    ep_r_hist = []
    plt.ion()
    for epoch in range(EPOCHS):
        s = env.reset()
        ep_r = 0
        while True:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            a_ = agent.choose_action(s_)
            agent.learn(s, a, r, s_, done, a_)

            ep_r += r
            total_r += r

            s = s_

            if done:
                break

        avg_r_epoch = total_r / (epoch + 1)
        ep_r_hist.append(avg_r_epoch)

        plt.cla()
        agent.plot_result(ep_r_hist, 'Epochs', 'Average Epoch Reward')
        plt.pause(0.0000001)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
