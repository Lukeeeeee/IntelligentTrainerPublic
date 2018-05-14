from src.agent.agent import Agent
import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
from src.config.config import Config
from config.key import CONFIG_KEY
from config import CONFIG


class BaselineTrainerAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/baselineTrainerAgentKey.json')

    def __init__(self, config, model, env):
        super(BaselineTrainerAgent, self).__init__(config=config, model=model, env=env)

    # def sample(self, path_nums, horizon):
    #
    #     obs = self.env.reset()
    #     action = self.predict(obs)
    #
    #     new_obs, reward, done, info = self.env.step(action)
    #
    #     # for i in range(self.config.config_dict['EPOCHS']):
    #     #     obs = self.env.reset()
    #     #     action = self.model.predict(obs)
    #     #     new_obs, reward, done, info = self.env.step(action)
    #     # pass

    def predict(self, state, *args, **kwargs):
        return np.array(self.model.predict(state))

    def init(self):
        self.model.init()
        super().init()


if __name__ == '__main__':
    conf = Config(standard_key_list=BaselineTrainerAgent.key_list)
    conf.load_config(path=CONFIG + '/baselineTrainerAgentTestConfig.json')
    a = BaselineTrainerAgent(config=conf, model=None, env=None)
