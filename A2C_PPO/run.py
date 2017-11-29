#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import PPO

GAME = 'Pendulum-v0'
EPISODES = 5000
MAX_STEP = 100

env = gym.make(GAME)
env = env.unwrapped
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
lowest_action = env.action_space.low
highest_action = env.action_space.high

agent = PPO(n_states, n_actions, low_action_bound=lowest_action, high_action_bound=highest_action,
            mode='CLIP', action_scale=2)


def run():
    plt.ion()
    total_r = 0
    avg_ep_r_hist = []
    for episode in range(EPISODES):
        ep_step = 0
        ep_r = 0
        s = env.reset()

        buffer_s, buffer_a, buffer_r = [], [], []

        while True:
            a = agent.predict_action(s)
            s_, r, done, info = env.step(a)
            done = True if ep_step >= 100 else False

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r+8)/8)

            ep_r += r
            total_r += r
            ep_step += 1

            if ep_step % agent.params_update_iter == 0 or done:
                v_ = agent.predict_value(s_)
                discounted_r = agent.discounted_reward(v_=v_, r=buffer_r)
                discounted_r = np.vstack(discounted_r)
                buffer_s = np.vstack(buffer_s)

                agent.learn(buffer_s, buffer_a, discounted_r)
                buffer_s, buffer_a, buffer_r = [], [], []

            s = s_

            if done:
                break

        avg_ep_r = total_r / (episode + 1)
        avg_ep_r_hist.append(avg_ep_r)
        if episode % 10 == 0:
            print('Episode %s Reward %s' % (episode, avg_ep_r))
        if episode >= 50:
            plt.cla()
            plt.plot(avg_ep_r_hist)
            plt.pause(0.001)
    plt.show()
    plt.ioff()


if __name__ == '__main__':
    run()
