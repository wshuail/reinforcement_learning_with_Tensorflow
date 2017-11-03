#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import matplotlib.pyplot as plt
from Double_DQN import DoubleDQN

EPISODES = 300
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

observation_space = env.observation_space.shape[0]
actions_space = env.action_space.n

agent = DoubleDQN(n_states=observation_space, n_actions=actions_space,
                  double_dqn=True)


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
            ep_r += r
            total_r += r

            agent.store_transition(s, a, r, s_)

            if agent.memory_counter >= agent.batch_size:
                agent.learn()

            s = s_

            if done:
                break

            step += 1

        avg_reward = total_r/(episode+1)
        ep_r_hist.append(avg_reward)

        if episode % 50:
            print('Episode %s Reward %s' % (episode, ep_r))

        plt.cla()
        agent.plot_result(data=ep_r_hist, x_label='Episodes', y_label='Reward')
        plt.pause(0.0000001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run()
