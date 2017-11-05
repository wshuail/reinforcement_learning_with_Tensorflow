#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from DDQN import DDQN

GAME = 'CartPole-v0'
EPOCHS = 3000

env = gym.make(GAME)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = DDQN(n_states, n_actions, priority=True, capacity=5000)


def run():
    plt.ion()
    total_r = 0
    total_step = 0
    ep_r_hist = []
    for epoch in range(EPOCHS):
        s = env.reset()
        ep_r = 0
        while True:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            r = -10 if done else r
            terminal = 0 if done else 1
            agent.store_transition(s, a, r, s_, terminal)

            ep_r += 1
            total_r += r
            total_step += 1

            if total_step >= agent.batch_size:
                agent.learn()

            s = s_

            if done:
                break

        if epoch % 50 == 0:
            print('Episode %s Reward %s' % (epoch, ep_r))

        if epoch >= 20:
            avg_ep_r = total_r / (epoch + 1)
            ep_r_hist.append(avg_ep_r)
            plt.cla()
            plt.plot(ep_r_hist)
            plt.pause(0.00001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
