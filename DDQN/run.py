#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
import numpy as np
from DQN import DQN

EPISODES = 3000
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

observation_space = env.observation_space.shape[0]
actions_space = env.action_space.n

agent = DQN(n_states=observation_space, n_actions=actions_space)


def run():

    step = 0
    ep_r_hist = []
    total_r = 0

    plt.ion()

    for episode in range(EPISODES):
        if RENDER:
            env.render()
        s = env.reset()
        ep_r = 0

        while True:

            a = agent.choose_action(s)

            s_, r, done, info = env.step(a)
            r = -10 if done else r
            terminal = 0 if done else 1

            ep_r += r
            total_r += r

            agent.store_transition(s, a, r, s_, terminal)

            if agent.memory_counter >= agent.batch_size:
                agent.learn()

            s = s_

            if done:
                break

            step += 1

        if episode % 50 == 0:
            print('Episode %s Reward %s' % (episode, ep_r))

        if episode > 100:
            avg_reward = total_r / (episode + 1)
            ep_r_hist.append(avg_reward)
            plt.cla()
            agent.plot_result(data=ep_r_hist, x_label='Episodes', y_label='Reward')
            plt.pause(0.0000001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
