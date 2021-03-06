from src.env.env import BasicEnv
from gym.spaces.box import Box
import numpy as np
from src.config.config import Config
from conf.key import CONFIG_KEY
from collections import deque
from src.agent.randomAgent.randomAgent import RandomAgent
from src.util.sampler.sampler import Sampler
import tensorflow as tf
from src.util.utils import DynamicsEnvironmentMemory
from src.env.util.baselineTrainerEnvStep import BaselineTrainerEnvStep


class BaselineTrainerEnv(BasicEnv):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/baselineTrainerEnvKey.json')

    def __init__(self, config, cyber_env, real_env, target_agent, test_env):
        super(BaselineTrainerEnv, self).__init__(config=config)
        self.action_space = Box(low=0,
                                high=1e10,
                                shape=np.array(self.config.config_dict['ACTION_SPACE']))

        self.observation_space = Box(low=-1e10,
                                     high=1e10,
                                     shape=np.array(self.config.config_dict['STATE_SPACE']))

        self.cyber_env = cyber_env
        self.real_env = real_env
        self.target_agent = target_agent
        self.test_env = test_env
        self.random_agent = RandomAgent(config=None, env=test_env, sampler=Sampler())
        self.stepper = BaselineTrainerEnvStep(config=None,
                                              registred_type=self.config.config_dict['BASELINE_ENV_STEP_TYPE'])

        self.real_env_sample_memory = DynamicsEnvironmentMemory()

        self.target_agent_real_env_reward_deque = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_REAL_ENV_REWARD_QUEUE_MAX_LENGTH'])

        self.target_agent_cyber_env_reward_deque = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_CYBER_ENV_REWARD_QUEUE_MAX_LENGTH'])
        self.dyna_error_dequeu = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_CYBER_ENV_REWARD_QUEUE_MAX_LENGTH'])

        self.critic_change = 0.
        self.actor_change = 0.
        self.critic_loss = 0.
        self.actor_loss = 0.
        self.sample_count = 0
        self.last_test = 0
        self.last_train = 0

    def step(self, action):
        return self.stepper.step(env=self, action=action)

    def reset(self):
        super().reset()
        return self.observation_space.sample()

    def init(self):
        self.cyber_env.init()
        self.cyber_env.reset()
        if hasattr(self.real_env, 'init') and callable(self.real_env.init):
            self.real_env.init()
        if hasattr(self.test_env, 'init') and callable(self.test_env.init):
            self.test_env.init()
        self.real_env.reset()
        self.test_env.reset()
        self.target_agent.init()
        self.init_train()
        super().init()

    def init_train(self):
        print("\nInit train----------------------")
        self.random_agent.sampler.env_status = self.random_agent.sampler.config.config_dict['REAL_ENVIRONMENT_STATUS']
        sample_data = self.random_agent.sample(env=self.real_env,
                                               sample_count=self.config.config_dict['CYBER_INIT_TRAIN_SAMPLE_COUNT'],
                                               store_flag=False,
                                               agent_print_log_flag=False)
        for j in range(len(sample_data.state_set)):
            data_dict = {
                'obs0': sample_data.state_set[j],
                'obs1': sample_data.new_state_set[j],
                'action': sample_data.action_set[j],
                'reward': sample_data.reward_set[j],
                'terminal1': sample_data.done_set[j],
                'delta_state': sample_data.new_state_set[j] - sample_data.state_set[j]
            }
            self.real_env_sample_memory.append(data_dict)
        self.cyber_env.model.update_mean_var(state_input=np.array(sample_data.state_set),
                                             action_input=np.array(sample_data.action_set),
                                             delta_state_label=np.array(sample_data.new_state_set) -
                                                               np.array(sample_data.state_set))
        for i in range(self.config.config_dict['DYNAMICS_TRAIN_ITERATION']):
            data = self.real_env_sample_memory.sample(batch_size=self.cyber_env.model.config.config_dict['BATCH_SIZE'])
            self.cyber_env.status = self.status_key['TRAIN']
            self.cyber_env.fit(state_set=data['obs0'],
                               action_set=data['action'],
                               delta_state_label_set=data['delta'],
                               sess=tf.get_default_session())
            self.cyber_env.print_log_queue(status=self.status_key['TRAIN'])
        self.real_env.reset()
        sample_data = self.target_agent.sample(env=self.real_env,
                                               sample_count=250,
                                               store_flag=False,
                                               agent_print_log_flag=False)
        unscaled_data = np.concatenate([[state.tolist() + [0]] for state in sample_data.state_set], axis=0)
        self.target_agent.model.update_scale(unscaled_data=unscaled_data)
        self.real_env.reset()

    def close(self):
        print('Close the trainer environment')

    def configure(self):
        pass

    def seed(self, seed=None):
        pass

    def get_state(self, env):
        return 0.

    def _sample_from_real_env(self, sample_count, sample_step):
        self.target_agent.status = self.target_agent.status_key['TRAIN']
        self.target_agent.env_status = self.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
        real_reward_data_this_step = []
        for i in range(sample_step):
            sample_data = self.target_agent.sample(env=self.real_env,
                                                   sample_count=sample_count,
                                                   store_flag=True,
                                                   agent_print_log_flag=True)

            for j in range(len(sample_data.state_set)):
                real_reward_data_this_step.append(sample_data.reward_set[j])

                data_dict = {
                    'obs0': sample_data.state_set[j],
                    'obs1': sample_data.new_state_set[j],
                    'action': sample_data.action_set[j],
                    'reward': sample_data.reward_set[j],
                    'terminal1': sample_data.done_set[j],
                    'delta_state': sample_data.new_state_set[j] - sample_data.state_set[j]
                }
                self.real_env_sample_memory.append(data_dict)
            self.cyber_env.model.update_mean_var(state_input=np.array(sample_data.state_set),
                                                 action_input=np.array(sample_data.action_set),
                                                 delta_state_label=np.array(sample_data.new_state_set) -
                                                                   np.array(sample_data.state_set))
        self.target_agent_real_env_reward_deque.append(np.mean(real_reward_data_this_step))

    def _sample_from_cyber_env(self, sample_count, sample_step):
        cyber_reward_data_this_step = []
        self.target_agent.env_status = self.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
        self.target_agent.status = self.target_agent.status_key['TRAIN']

        for i in range(sample_step):
            sample_data = self.target_agent.sample(env=self.cyber_env,
                                                   sample_count=sample_count,
                                                   store_flag=True,
                                                   agent_print_log_flag=True)
            for j in range(len(sample_data.state_set)):
                cyber_reward_data_this_step.append(sample_data.reward_set[j])
        self.target_agent_cyber_env_reward_deque.append(np.mean(cyber_reward_data_this_step))
