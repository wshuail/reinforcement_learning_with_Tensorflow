#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from Actor_Critic import Actor, Critic

GAME = 'CartPole-v0'
EPISODES = 1000
RENDER = False

env = gym.make(GAME)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = Actor(n_states, n_actions)
critic = Critic(n_states)


def run():
    plt.ion()
    total_r = 0
    avg_ep_r_hist = []
    for episode in range(EPISODES):
        ep_r = 0
        s = env.reset()
        while True:
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            r = -10 if done else r

            ep_r += r
            total_r += r

            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)

            s = s_

            if done:
                if episode % 30 == 0:
                    print('Episode %d Reward %s' %(episode, ep_r))
                break
        if episode >= 20:
            avg_ep_r = total_r/(episode+1)
            avg_ep_r_hist.append(avg_ep_r)
        plt.cla()
        plt.plot(avg_ep_r_hist)
        plt.pause(0.0001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
