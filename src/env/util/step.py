from src.core import Basic
import tensorflow as tf
import numpy as np


class Step(Basic):

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, env, action):
        raise NotImplementedError

    def step(self, env, action):
        env.step_count += 1
