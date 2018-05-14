from src.env.env import BasicEnv
from gym.spaces.box import Box
import numpy as np
from src.config.config import Config
from config.key import CONFIG_KEY
from collections import deque
from src.agent.randomAgent.randomAgent import RandomAgent
from src.util.sampler.sampler import Sampler
import tensorflow as tf
from src.env.utils import DynamicsEnvironmentMemory


class BaselineTrainerEnv(BasicEnv):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/baselineTrainerEnvKey.json')

    def __init__(self, config, cyber_env, real_env, target_agent, test_env):
        super(BaselineTrainerEnv, self).__init__(config=config)
        # TODO DESIGN THE HIGH AND LOW FOR ACTION AND OBS SPACE
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

        self.real_env_sample_memory = DynamicsEnvironmentMemory()

        self.target_agent_real_env_reward_deque = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_REAL_ENV_REWARD_QUEUE_MAX_LENGTH'])

        self.target_agent_cyber_env_reward_deque = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_CYBER_ENV_REWARD_QUEUE_MAX_LENGTH'])
        self.dyna_error_dequeu = \
            deque(maxlen=self.config.config_dict['TARGET_AGENT_CYBER_ENV_REWARD_QUEUE_MAX_LENGTH'])
        self.critic_change = 0.0
        self.actor_change = 0.0
        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.sample_count = 0
        self.last_test = 0
        self.last_train = 0

    def step(self, action):
        super().step(action=action)

        F1 = action[0]
        prob_sample_on_real = action[1]
        prob_train_on_real = action[2]
        # spare_action = action[3]

        self.target_agent.sampler.set_F(F1=F1, F2=0.0)

        print("\n Real Env used count %d" % self.target_agent._real_env_sample_count)

        print("\nSample for target agent----------------------")

        self.target_agent.status = self.target_agent.status_key['TRAIN']
        self.target_agent.env_status = self.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
        # self.target_agent.log_queue.queue.clear()
        real_reward_data_this_step = []
        K_r = self.config.config_dict['SAMPLE_COUNT_PER_STEP']
        # TODO
        # TOO UGLY OF THIS JUDGE!!!
        from src.util.sampler.fakeSampler import FakeSampler
        from src.util.sampler.fakeIntelligentSampler import FakeIntelligentSampler
        if isinstance(self.target_agent.sampler, (FakeSampler, FakeIntelligentSampler)):
            sample_count = K_r
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_r
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

        cyber_reward_data_this_step = []
        K_c = int(
            self.config.config_dict['SAMPLE_COUNT_PER_STEP'] / prob_sample_on_real * (1.0 - prob_sample_on_real))
        if K_c < 1:
            K_c = 1

        if isinstance(self.target_agent.sampler, FakeSampler):
            sample_count = K_c
            sample_step = 1
        else:
            sample_count = 1
            sample_step = K_c

        self.target_agent.env_status = self.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
        self.target_agent.status = self.target_agent.status_key['TRAIN']
        # self.target_agent.log_queue.queue.clear()
        for i in range(sample_step):
            sample_data = self.target_agent.sample(env=self.cyber_env,
                                                   sample_count=sample_count,
                                                   store_flag=True,
                                                   agent_print_log_flag=True)
            for j in range(len(sample_data.state_set)):
                cyber_reward_data_this_step.append(sample_data.reward_set[j])
        self.target_agent_cyber_env_reward_deque.append(np.mean(cyber_reward_data_this_step))

        # self.target_agent.print_log_queue(status=self.status_key['TRAIN'])

        self.sample_count += self.config.config_dict['SAMPLE_COUNT_PER_STEP']

        print("\nTrain for target agent from real env----------------------")
        self.target_agent.status = self.status_key['TRAIN']

        critic_loss_set = []
        actor_loss_set = []

        t_r = int(max(self.config.config_dict['TARGET_AGENT_TRAIN_ITERATION'] * 0.6, 1))
        t_c = int((1 - prob_train_on_real) * t_r / prob_train_on_real)

        total_train = int(t_r + t_c)

        for i in range(total_train):
            prob = np.random.rand()
            if prob <= prob_train_on_real:
                self.target_agent.env_status = self.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
                res_dict = self.target_agent.train()
            else:
                self.target_agent.env_status = self.target_agent.config.config_dict['CYBER_ENVIRONMENT_STATUS']
                res_dict = self.target_agent.train()

            self.target_agent.print_log_queue(status=self.status_key['TRAIN'])

            if res_dict is not None:
                critic_loss_set.append(res_dict['VALUE_FUNCTION_LOSS'])
                actor_loss_set.append(res_dict['CONTROLLER_LOSS'])
            else:
                critic_loss_set.append(0.0)
                actor_loss_set.append(0.0)
        self.critic_change = critic_loss_set[-1] + (critic_loss_set[0] - critic_loss_set[-1]) / (
                np.std(critic_loss_set) + 0.0001)
        self.actor_change = actor_loss_set[-1] + (actor_loss_set[0] - actor_loss_set[-1]) / (
                np.std(actor_loss_set) + 0.0001)

        new_critic_loss = np.mean(critic_loss_set)
        new_actor_loss = np.mean(actor_loss_set)
        self.critic_change += self.critic_loss - new_critic_loss
        self.actor_change += self.actor_loss - new_actor_loss
        self.critic_loss = new_critic_loss
        self.actor_loss = new_actor_loss

        final_step_dynamics_train_loss = -1

        print("\nTrain for dynamics env----------------------")

        self.cyber_env.status = self.status_key['TRAIN']

        for i in range(self.config.config_dict['DYNAMICS_TRAIN_ITERATION']):
            data = self.real_env_sample_memory.sample(
                batch_size=self.cyber_env.model.config.config_dict['BATCH_SIZE'])

            final_step_dynamics_train_loss = self.cyber_env.fit(state_set=data['obs0'],
                                                                action_set=data['action'],
                                                                delta_state_label_set=data['delta'],
                                                                sess=tf.get_default_session())

            self.cyber_env.print_log_queue(self.status_key['TRAIN'])
        self.dyna_error_dequeu.append(final_step_dynamics_train_loss)

        progress_bar = np.floor(
            1.0 * self.target_agent._real_env_sample_count / self.config.config_dict['TEST_FRIQUENCY_SAMPLE'])
        if progress_bar > self.last_test:
            self.last_test = progress_bar
            print("\nTest for dynamics env----------------------")
            self.target_agent.status = self.status_key['TEST']
            self.target_agent.env_status = self.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
            self.cyber_env.status = self.status_key['TEST']
            if "DYNAMICS_ENV_TEST_SAMPLE" in self.config.config_dict is False:
                self.config.config_dict['DYNAMICS_ENV_TEST_SAMPLE'] = 1000
            sample_data = self.target_agent.sample(env=self.test_env,
                                                   sample_count=self.config.config_dict['DYNAMICS_ENV_TEST_SAMPLE'],
                                                   store_flag=False,
                                                   agent_print_log_flag=False)

            self.cyber_env.test(state_set=sample_data.state_set,
                                action_set=sample_data.action_set,
                                delta_state_label_set=np.array(sample_data.new_state_set) - np.array(
                                    sample_data.state_set),
                                sess=tf.get_default_session())
            self.cyber_env.print_log_queue(status=self.status_key['TEST'])

            print("\nTest for target agent by real cost function----------------------")
            self.target_agent.status = self.status_key['TEST']
            self.target_agent.env_status = self.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']

            assert self.target_agent.sampler.env_status == \
                   self.target_agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']

            sample_data = self.target_agent.sample(env=self.test_env,
                                                   sample_count=self.config.config_dict['TARGET_AGENT_TEST_SAMPLE'],
                                                   store_flag=False,
                                                   agent_print_log_flag=True)

        obs = 0.0
        reward = 0.0
        done = False
        info = [self.target_agent._real_env_sample_count, ]
        print("EEE self.target_agent._real_env_sample_count=", self.target_agent._real_env_sample_count)
        return obs, reward, done, info

    def reset(self):
        super().reset()
        return self.observation_space.sample()

    def init(self):
        # Store some init data for dyna env
        self.cyber_env.init()
        self.cyber_env.reset()
        if hasattr(self.real_env, 'init') and callable(self.real_env.init):
            self.real_env.init()
        if hasattr(self.test_env, 'init') and callable(self.test_env.init):
            self.test_env.init()
        self.real_env.reset()
        self.test_env.reset()
        self.target_agent.init()
        # TODO !
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
                                               store_flag=True,
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
        return None
