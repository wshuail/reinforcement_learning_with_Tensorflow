#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from Actor_Critic import Actor, Critic

GAME = 'Pendulum-v0'
EPISODES = 5000
MAX_STEP = 100

env = gym.make(GAME)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
low_action_bound = env.action_space.low[0]
high_action_bound = env.action_space.high[0]

actor = Actor(n_states, n_actions, low_action_bound, high_action_bound)
critic = Critic(n_states)


def run():
    plt.ion()
    total_r = 0
    avg_ep_r_hist = []
    for episode in range(EPISODES):
        ep_step = 0
        s = env.reset()
        while True:
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)

            total_r += r
            ep_step += 1

            td_error = critic.learn(s, (r+8)/8, s_)
            actor.learn(s, a, td_error)

            s = s_

            if ep_step >= MAX_STEP:
                break
        avg_ep_r = total_r / (episode + 1)
        if episode % 20 == 0:
            print('Episode %d Avg Reward %s' %(episode, avg_ep_r))
        if episode >= 50:
            # print('Episode %s Reward %s' % (episode, avg_ep_r))
            avg_ep_r_hist.append(avg_ep_r)
            plt.cla()
            plt.plot(avg_ep_r_hist)
            plt.pause(0.0001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
