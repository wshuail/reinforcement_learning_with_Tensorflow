#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from Policy_Gradient import PolicyGradient

EPISODES = 5000

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

observation_space = env.observation_space.shape[0]
actions_space = env.action_space.n

agent = PolicyGradient(n_states=observation_space, n_actions=actions_space,
                       gamma=0.99)


def run():
    step = 0
    total_r = 0
    avg_ep_r_hist = []
    plt.ion()
    for episode in range(EPISODES):
        s = env.reset()

        ep_r = 0

        while True:

            a = agent.choose_action(s)

            s_, r, done, info = env.step(a)
            r = -10 if done else r

            agent.store_transition(s, a, r)

            ep_r += r
            total_r += r

            if done:
                agent.learn()
                break

            s = s_
            step += 1

        if episode % 30 == 0:
            print('Episode %s Reward %s' % (episode, ep_r))

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
