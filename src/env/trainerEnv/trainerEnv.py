from src.env.env import BasicEnv
from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
from gym.spaces.box import Box
import numpy as np
from src.config.config import Config
from config import CONFIG
from config.key import CONFIG_KEY
import tensorflow as tf
from collections import deque
from scipy.optimize import curve_fit
from statsmodels.tsa.arima_model import AR
import pandas as pd


def fit_loss(y_data):
    # data = pd.Series(y_data, index=[str(i) for i in range(len(y_data))])
    data = y_data
    model = AR(data)
    res = model.fit()
    ans = res.predict(len(y_data) - 1, len(y_data), dynamic=True)[-1]
    return ans


class TrainerEnv(BaselineTrainerEnv):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/trainerEnvKey.json')

    def __init__(self, config, cyber_env, real_env, target_agent, test_env):
        super(TrainerEnv, self).__init__(config=config,
                                         cyber_env=cyber_env,
                                         real_env=real_env,
                                         target_agent=target_agent,
                                         test_env=test_env)

        high = np.ones([3, ])
        low = 0.2 * high

        self.action_space = Box(low=low,
                                high=high)

        self.observation_space = Box(low=-1e10,
                                     high=1e10,
                                     shape=np.array(self.config.config_dict['STATE_SPACE']))

        self.td = self.config.config_dict['TD']

        self.real_r_his = [-100.0, -100.0, -100.0]
        self.cyber_r_his = [-100.0, -100.0, -100.0]
        self.dyna_err_his = [-100.0, -100.0, -100.0]
        self.status = self.status_key['TRAIN']

    def step(self, action):
        print("Trainer action=", action)
        super().step(action=action)
        # TODO USE THE VALUE TO GENERATE YOUR OWN OBS, ACTION ..
        ln = len(self.target_agent_real_env_reward_deque)
        real_r_his = []
        cyber_r_his = []
        dyna_err_his = []
        for i in range(self.td):
            idx = ln - (self.td - i)
            if idx < 0:
                idx = 0
            real_r_his.append(self.target_agent_real_env_reward_deque[idx])
            cyber_r_his.append(self.target_agent_cyber_env_reward_deque[idx])
            dyna_err_his.append(self.dyna_error_dequeu[idx])

        self.real_r_his = self.real_r_his + real_r_his
        self.cyber_r_his = cyber_r_his
        self.dyna_err_his = dyna_err_his
        obs = self._get_obs()
        reward = self._get_reward(action)

        done = False
        info = [0.0]
        print("Trainer reward=", reward)
        self.log_file_content.append({
            'INDEX': self.log_print_count,
            'OBS': np.array(obs).squeeze().tolist(),
            'REWARD': float(reward),
            'DONE': done,
            'ACTION': np.array(action).squeeze().tolist(),
            'VALUE_FUNCTION_LOSS': self.critic_loss,
            'CONTROLLER_LOSS': self.actor_loss,
            'VALUE_FUNCTION_LOSS_CHANGE': self.critic_change,
            'CONTROLLER_LOSS_CHANGE': self.actor_change
        })
        self.log_print_count += 1
        return obs, reward, done, info

    def _get_reward(self, action):
        ####concate the useful training info from last td steps

        # re = np.sign(self.real_r_his[-1] - self.real_r_his[
        #     -2])  # +(real_r_his[-1]-real_r_his[-3]) #+0.00001*(cyber_r_his[-1]-cyber_r_his[-2]) + 0.0001*dyna_err_his[-1]
        # re += self.critic_change
        # re += self.actor_change
        # if (self.cyber_r_his[-1]-self.real_r_his[-1])**2/(self.real_r_his[-1])**2>0.5:
        #     re -=(2-action[1]-action[2])*(self.cyber_r_his[-1]-self.real_r_his[-1])**2

        # reward_len = len(self.real_r_his)
        #
        # if reward_len > 100:
        #     self.real_r_his = self.real_r_his[-100:]
        #     reward_len = len(self.real_r_his)
        # if reward_len - 1 < 50:
        #     re = np.sign(self.real_r_his[-1] - self.real_r_his[-2])
        # else:
        #     y_data = [self.real_r_his[i] for i in range(reward_len - 1)]
        #     pred_y = fit_loss(y_data=y_data)
        #     re = np.sign(self.real_r_his[-1] - pred_y)

        re = np.sign(self.real_r_his[-1] - self.real_r_his[-2])

        print("re=", re)
        return re

    def _get_obs(self):

        re = [np.sign(self.real_r_his[-1] - self.real_r_his[-2]), np.sign(self.real_r_his[-2] - self.real_r_his[-3]),
              self.dyna_err_his[
                  -1]]  # [np.sign(self.real_r_his[-1]-self.real_r_his[-2]), np.sign(self.critic_change), np.sign(self.actor_change),]
        re = [self.dyna_err_his[-1]]
        re = np.asarray(re)
        return self.real_r_his[-1]  # re #re #self.target_agent._real_env_sample_count

    def reset(self):
        super().reset()

    def init(self):
        super().init()

    def close(self):
        super().close()

    def get_state(self, env):
        return self._get_obs()
