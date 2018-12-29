from src.core import Basic
import numpy as np
import math


class TrainerEnvReward(Basic):
    def __init__(self, config, registred_type):
        super().__init__(config)
        if registred_type == 'SGN_LAST_TWO_REWARD':
            self.reward = self.sgn_differ_last_two_real_sample_reward
        elif registred_type == 'DIFF_LAST_TWO_REWARD':
            self.reward = self.differ_last_two_real_sample_reward
        elif registred_type == 'LAST_ONE_REWARD':
            self.reward = self.last_reward_sample_reward
        elif registred_type == 'DIIFER_REAL_CYBER_REWARD':
            self.reward = self.differ_real_cyber_sample_reward
        elif registred_type == 'V4_REWARD':
            self.reward = self.v4_reward
        else:
            raise IndexError('%s type of reward did not exists' % registred_type)

    def __call__(self, env, action=None):
        return self.reward(env, action)

    @staticmethod
    def sgn_differ_last_two_real_sample_reward(env, action=None):
        reward = np.sign(env.real_r_his[-1] - env.real_r_his[-2])
        return reward

    @staticmethod
    def last_reward_sample_reward(env, action=None):
        reward = env.real_r_his[-1]
        return reward

    @staticmethod
    def differ_last_two_real_sample_reward(env, action=None):
        reward = env.real_r_his[-1] - env.real_r_his[-2]
        return reward

    @staticmethod
    def differ_real_cyber_sample_reward(env, action):
        reward = -math.fabs(env.real_r_his[-1] - env.cyber_r_his[-1]) * \
                 action[1] * (1 - action[1] + 0.0001) * 100
        return reward

    @staticmethod
    def v4_reward(env, action=None):
        reward = env.target_agent_real_env_reward_deque[-1], env.target_agent_real_env_reward_deque[-2]
        return reward
