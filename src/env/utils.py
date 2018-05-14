import numpy as np
import json
from src.config.config import Config
from config.costFunction import COST_FUNCTION_PATH
from sklearn.preprocessing import PolynomialFeatures
import math
import gym

from collections import deque
import numpy as np

ENV_MAX_EPOSIDE_STEPS = {
    "MountainCarContinuous-v0": 999,
    "Pendulum-v0": 200,
    "Ant-v1": 1000,
    "HumanoidStandup-v1": 1000,
    "Swimmer-v1": 1000,
    "HalfCheetah": 1000,
    "Reacher-v1": 50
}


class EnvMakerWrapper(object):
    def __init__(self, name):
        self.id = name
        self.name = name
        self.__class__.__name__ = self.id

    def __call__(self, *args, **kwargs):
        return gym.make(self.id)


from src.env.halfCheetahEnv.halfCheetahEnv import HalfCheetahEnvNew

GAME_ENV_NAME_DICT = {
    "Pendulum-v0": EnvMakerWrapper('Pendulum-v0'),
    "Ant-v1": EnvMakerWrapper('Ant-v1'),
    "HumanoidStandup-v1": EnvMakerWrapper('HumanoidStandup-v1'),
    "MountainCarContinuous-v0": EnvMakerWrapper('MountainCarContinuous-v0'),
    "Swimmer-v1": EnvMakerWrapper('Swimmer-v1'),
    "HalfCheetah": HalfCheetahEnvNew,
    "Reacher-v1": EnvMakerWrapper('Reacher-v1'),
}

pendulum_instance = GAME_ENV_NAME_DICT['Pendulum-v0']()
ant_instance = GAME_ENV_NAME_DICT['Ant-v1']()
humanoid_standup_instance = GAME_ENV_NAME_DICT['HumanoidStandup-v1']()
mountain_car_continuous_instance = GAME_ENV_NAME_DICT['MountainCarContinuous-v0']()
swimmer_instance = GAME_ENV_NAME_DICT['Swimmer-v1']()
half_cheetah_instance = GAME_ENV_NAME_DICT['HalfCheetah']()
reacher_instance = GAME_ENV_NAME_DICT['Reacher-v1']()


def pendulum_get_state(env):
    if isinstance(env, type(pendulum_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def ant_get_state(env):
    if isinstance(env, type(ant_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def humanoid_standup_get_state(env):
    if isinstance(env, type(humanoid_standup_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def mountain_car_continuous_get_state(env):
    if isinstance(env, type(mountain_car_continuous_instance)) is True:
        return env.unwrapped.state
    else:
        raise ValueError('Wrong type of environment to get state')


def swimmer_get_state(env):
    if isinstance(env, type(swimmer_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def half_cheetah_get_state(env):
    if isinstance(env, type(half_cheetah_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def reacher_get_state(env):
    if isinstance(env, type(reacher_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


GET_STATE_FUNCTION_DICT = {
    "Pendulum-v0": pendulum_get_state,
    "Ant-v1": ant_get_state,
    "HumanoidStandup-v1": humanoid_standup_get_state,
    "MountainCarContinuous-v0": mountain_car_continuous_get_state,
    "Swimmer-v1": swimmer_get_state,
    "HalfCheetah": half_cheetah_get_state,
    "Reacher-v1": reacher_get_state
}


def make_env(id):
    env = GAME_ENV_NAME_DICT[id]()
    env.get_state = GET_STATE_FUNCTION_DICT[id]
    return env


class DynamicsEnvironmentMemory(object):
    def __init__(self):
        self.data = deque(maxlen=2000)

    def sample(self, batch_size):
        batch_size = min(len(self.data), batch_size)
        batch_idxs = np.random.random_integers(len(self.data) - 2, size=batch_size)

        batch_set = []
        for idx in batch_idxs:
            batch_set.append(self.data[idx])
        obs0 = []
        obs1 = []
        reward = []
        action = []
        done = []
        delta = []
        for data in batch_set:
            obs0.append(data['obs0'])
            obs1.append(data['obs1'])
            reward.append(data['reward'])
            action.append(data['action'])
            done.append(data['terminal1'])
            delta.append(data['delta_state'])
        return {
            'obs0': obs0,
            'obs1': obs1,
            'action': action,
            'reward': reward,
            'terminal1': done,
            'delta': delta
        }

    def append(self, sample):
        self.data.append(sample)
