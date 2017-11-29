#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 18:22
# @Author  : Wang Shuailong

from __future__ import print_function
import multiprocessing
import threading
import tensorflow as tf
from brain import Brain
from agent import Agent
from environment import Environment

GAME = 'Pendulum-v0'


def run():
    sess = tf.Session()

    with tf.device("/cpu:0"):
        env = Environment(GAME)
        Brain('global', sess, env=env)
        workers = []
        cpu_number = multiprocessing.cpu_count()
        for i in range(cpu_number-2):
            env = Environment(GAME)
            workers.append(Agent(i, sess, env))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)


if __name__ == '__main__':
    run()