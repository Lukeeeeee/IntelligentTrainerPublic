from src.agent.agent import Agent
import numpy as np


class RandomAgent(Agent):

    def __init__(self, config, env, sampler, model=None):
        super(RandomAgent, self).__init__(config=config, env=env, model=model, sampler=sampler)

    def predict(self, state, *args, **kwargs):
        return np.array(self.env.action_space.sample())

    def print_log_queue(self, status):
        self.status = status
        pass
