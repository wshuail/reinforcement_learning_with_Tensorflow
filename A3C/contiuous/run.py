#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import threading
import tensorflow as tf
from brain import Brain
from agent import Agent
from environment import Environment

GAME = 'Pendulum-v0'


def run():
    coord = tf.train.Coordinator()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    with tf.device('/cpu:0'):
        Brain(scope='global', env=Environment(name=GAME))

        agents = []
        cpu_number = multiprocessing.cpu_count()
        for i in range(cpu_number-2):
            env = Environment(name=GAME)
            agent = Agent(number=i, env=env, coordinator=coord)
            agents.append(agent)

    agent_threads = []

    for agent in agents:
        job = lambda: agent.work()
        t = threading.Thread(target=job)
        t.start()

        agent_threads.append(t)
    coord.join(agent_threads)


if __name__ == '__main__':
    run()
