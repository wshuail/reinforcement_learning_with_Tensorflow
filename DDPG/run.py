#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from ddpg import DDPG

GAME = 'Pendulum-v0'
MAX_STEP = 100
EPISODES = 3000


env = gym.make(GAME)
env = env.unwrapped
env.seed(1)

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
action_low_bound = env.action_space.low
action_high_bound = env.action_space.high

agent = DDPG(n_states, n_actions, action_low_bound, action_high_bound)


def run():
    plt.ion()
    total_r = 0
    avg_ep_r_hist = []
    for episode in range(EPISODES):
        ep_step = 0
        ep_r = 0
        s = env.reset()
        while True:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            agent.store_memory(s, a, r, s_)

            ep_r += r
            total_r += r
            ep_step += 1

            if agent.memory_counter >= agent.batch_size:
                agent.learn()

            if ep_step >= MAX_STEP:
                break

            s = s_

        if episode >= 10:
            avg_ep_r = total_r/(episode+1)
            avg_ep_r_hist.append(avg_ep_r)
            if episode % 20 == 0:
                print('Episode %d Avg Reward/Ep %s' % (episode, avg_ep_r))
        plt.cla()
        plt.plot(avg_ep_r_hist)
        plt.pause(0.0001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()

