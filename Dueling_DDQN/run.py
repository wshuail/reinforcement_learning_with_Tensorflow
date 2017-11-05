#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from Dueling_DDQN import DDQN

EPOCHS = 1000

GAME = 'CartPole-v0'
env = gym.make(GAME)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = DDQN(n_states, n_actions)


def run():
    plt.ion()
    avg_ep_r_hist = []
    total_r = 0
    for epoch in range(EPOCHS):
        ep_r = 0
        s = env.reset()
        while True:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            terminal = 0 if done else 1
            r = -10 if done else r

            agent.store_transition(s, a, r, s_, terminal)

            ep_r += r
            total_r += r

            if agent.memory_counter >= agent.batch_size:
                agent.learn()

            s = s_

            if done:
                break

        if epoch % 50 == 0:
            print('Epoch %d Reward %s' %(epoch, ep_r))

        if epoch >= 10:
            avg_ep_r = total_r/epoch
            avg_ep_r_hist.append(avg_ep_r)
            plt.cla()
            plt.plot(avg_ep_r_hist)
            plt.xlabel('Epoch')
            plt.ylabel('Average Reward Per Epoch')
            plt.pause(0.00001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()