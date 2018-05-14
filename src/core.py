import tensorflow as tf
import time
from log.baselineTestLog import LOG
import os
from src.config.config import Config
from config.key import CONFIG_KEY
import numpy as np
import random
import queue
import json
import easy_tf_log
from log.intelligentTestLog import INTEL_LOG


class Basic(object):
    key_list = []
    status_key = {'TRAIN': 0, 'TEST': 1}

    def __init__(self, config):
        self.config = config
        self.name = type(self).__name__

        self._train_log_file = self.name + '_train_.log'
        self._test_log_file = self.name + '_test_.log'

        self._train_log_queue = queue.Queue(maxsize=1e10)
        self._test_log_queue = queue.Queue(maxsize=1e10)

        self._train_log_print_count = 0
        self._test_log_print_count = 0

        self._train_log_file_content = []
        self._test_log_file_content = []

        self._status = Basic.status_key['TRAIN']

        self._log_file = None
        self._log_queue = None
        self._log_print_count = None
        self._log_file_content = None

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            content = self.log_queue.get()
            self.log_file_content.append({'INDEX': self.log_print_count, 'LOG': content})
            print(content)
            self.log_print_count += 1

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_value):
        if new_value != Basic.status_key['TRAIN'] and new_value != Basic.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)

        if self._status == new_value:
            return
        self._status = new_value

    @property
    def log_file(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_queue(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_queue
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_queue
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_file_content(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file_content
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file_content
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_print_count(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_print_count
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_print_count
        raise KeyError('Current Status: %d did not existed' % self._status)

    @log_print_count.setter
    def log_print_count(self, new_val):
        if self._status == Basic.status_key['TRAIN']:
            self._train_log_print_count = new_val
        elif self._status == Basic.status_key['TEST']:
            self._test_log_print_count = new_val
        else:
            raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def current_status(self):
        if self._status == Basic.status_key['TRAIN']:
            return 'TRAIN'
        elif self._status == Basic.status_key['TEST']:
            return 'TEST'


class Logger(object):

    def __init__(self, prefix=None, log=LOG, log_path=None):
        if log_path is not None:
            self._log_dir = log_path
        else:
            self._log_dir = log + '/' + prefix + '/' + time.strftime("%Y-%m-%d_%H-%M-%S")
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        if os.path.exists(self._log_dir):
            raise FileExistsError('%s path is existed' % self._log_dir)
        self.tf_log = easy_tf_log
        self.tf_log.set_dir(log_dir=self._log_dir + '/tf/')

    @property
    def log_dir(self):
        if os.path.exists(self._log_dir) is False:
            os.makedirs(self._log_dir)
        return self._log_dir

    @property
    def config_file_log_dir(self):
        self._config_file_log_dir = os.path.join(self.log_dir, 'config')
        if os.path.exists(self._config_file_log_dir) is False:
            os.makedirs(self._config_file_log_dir)
        return self._config_file_log_dir

    @property
    def loss_file_log_dir(self):
        self._loss_file_log_dir = os.path.join(self.log_dir, 'loss')
        if os.path.exists(self._loss_file_log_dir) is False:
            os.makedirs(self._loss_file_log_dir)
        return self._loss_file_log_dir

    @property
    def model_file_log_dir(self):
        self._model_file_log_dir = os.path.join(self.log_dir, 'model/')
        if os.path.exists(self._model_file_log_dir) is False:
            os.makedirs(self._model_file_log_dir)
        return self._model_file_log_dir

    def out_to_file(self, file_path, content):
        with open(file_path, 'w') as f:
            # TODO how to modify this part
            for dict_i in content:
                for key, value in dict_i.items():
                    if isinstance(value, np.generic):
                        dict_i[key] = value.item()
            json.dump(content, fp=f, indent=4)


class GamePlayer(object):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/gamePlayerKey.json')

    def __init__(self, config, agent, env, basic_list, ep_type=0, log_path=None):
        self.config = config
        self.agent = agent
        self.env = env
        self.basic_list = []
        for basic in basic_list:
            if basic is not None:
                self.basic_list.append(basic)
        if ep_type == 1:
            self.logger = Logger(prefix=self.config.config_dict['GAME_NAME'], log=INTEL_LOG, log_path=log_path)
        else:
            self.logger = Logger(prefix=self.config.config_dict['GAME_NAME'], log=LOG, log_path=log_path)

    def set_seed(self, seed=None):
        if seed is None:
            seed = int(self.config.config_dict['SEED'])
        else:
            self.config.config_dict['SEED'] = seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

    def save_config(self):
        if self.config.config_dict['SAVE_CONFIG_FILE_FLAG'] == 1:
            for basic in self.basic_list:
                basic.config.save_config(path=self.logger.config_file_log_dir,
                                         name=basic.name + '.json')
            self.config.save_config(path=self.logger.config_file_log_dir,
                                    name='GamePlayer.json')

    def init(self):
        self.agent.init()
        self.env.init()

    def step(self):
        trainer_data = self.agent.sample(env=self.env,
                                         sample_count=1,
                                         store_flag=True,
                                         agent_print_log_flag=True)
        self.agent.update()
        return trainer_data

    def play(self, seed_new=None):
        self.set_seed(seed_new)
        self.save_config()
        self.init()

        info_set = []

        # TODO modify here to control the whole training process
        for i in range(self.config.config_dict['EPOCH']):
            for j in range(self.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                trainer_data = self.step()
                # info_set[0].append(self.agent.sampler.info[0])
                # info_set[1].append(self.agent.sampler.info[1])
                # info_set[2].append(self.agent.sampler.info[2])
                info_set.append(trainer_data)
                print("self.config.config_dict['MAX_REAL_ENV_SAMPLE']=", self.config.config_dict['MAX_REAL_ENV_SAMPLE'])
                print("self.env.target_agent._real_env_sample_count=", self.env.target_agent._real_env_sample_count)
                if self.env.target_agent._real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                    break
            if self.env.target_agent._real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                break
        # END
        return info_set

    def print_log_to_file(self):
        for basic in self.basic_list:
            if 'LOG_FLAG' in basic.config.config_dict and basic.config.config_dict['LOG_FLAG'] == 1:
                basic.status = basic.status_key['TRAIN']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)
                basic.status = basic.status_key['TEST']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)

    def save_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.save_model(path=self.logger.model_file_log_dir, global_step=1)

    def load_all_model(self):
        from src.model.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.load_model(path=self.logger.model_file_log_dir, global_step=1)


class AssembleGamePlayer(object):
    def __init__(self, intel_player, ref_player_list):
        self.main_player = intel_player
        self.reference_players = ref_player_list
        pass

    def play(self, seed_new=None):
        self.main_player.set_seed(seed_new)
        for player in self.reference_players:
            player.set_seed(seed_new)

        self.main_player.save_config()
        for player in self.reference_players:
            player.save_config()

        self.main_player.init()
        for player in self.reference_players:
            player.init()
        for i in range(self.main_player.config.config_dict['EPOCH']):
            for j in range(self.main_player.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                self.main_player.step()
                for player in self.reference_players:
                    player.step()
                print("self.config.config_dict['MAX_REAL_ENV_SAMPLE']=",
                      self.main_player.config.config_dict['MAX_REAL_ENV_SAMPLE'])
                print("self.env.target_agent._real_env_sample_count=",
                      self.main_player.env.target_agent._real_env_sample_count)
                if self.main_player.env.target_agent._real_env_sample_count + self.main_player.env.config.config_dict[
                    'SAMPLE_COUNT_PER_STEP'] > self.main_player.config.config_dict[
                    'MAX_REAL_ENV_SAMPLE'] * self.main_player.config.config_dict['REAL_ENV_SAMPLE_TRAIN_RATION']:
                    break
            if self.main_player.env.target_agent._real_env_sample_count + self.main_player.env.config.config_dict[
                'SAMPLE_COUNT_PER_STEP'] > self.main_player.config.config_dict[
                'MAX_REAL_ENV_SAMPLE'] * self.main_player.config.config_dict['REAL_ENV_SAMPLE_TRAIN_RATION']:
                break
        self.print_log_to_file()
        self.save_all_model()
        self.final_test_process()

    def print_log_to_file(self):
        self.main_player.print_log_to_file()
        for player in self.reference_players:
            player.print_log_to_file()

    def save_all_model(self):
        self.main_player.save_all_model()
        for player in self.reference_players:
            player.save_all_model()

    def final_test_process(self):
        test_sample_real_env_count = self.main_player.config.config_dict['MAX_REAL_ENV_SAMPLE'] - \
                                     self.main_player.env.target_agent._real_env_sample_count
        # test_sample_real_env_count = self.main_player.config.config_dict['MAX_REAL_ENV_SAMPLE'] * (
        #         1.0 - self.main_player.config.config_dict['REAL_ENV_SAMPLE_TRAIN_RATION'])
        target_agent_count = 1 + len(self.reference_players)
        test_sample_per_target_agent = int(test_sample_real_env_count // target_agent_count)
        # test_sample_per_target_agent = 1000
        # agent_dict = {
        #     self.main_player.env.target_agent.config.config_dict['NAME']: self.main_player.env.target_agent,
        # }
        # for player in self.reference_players:
        #     agent_dict[player.env.target_agent.config.config_dict['NAME']] = player.env.target_agent
        #
        # for name, target_agent in agent_dict:
        #     self._test_agent(target_agent=target_agent, test_sample_per_agent=test_sample_per_target_agent)
        best_reward = -1000000.0
        best_player = None
        print('Real env used %d Test sample for each player %d\n' % (
            self.main_player.env.target_agent._real_env_sample_count, test_sample_per_target_agent))
        reward = self._test_agent(env=self.main_player.env.test_env,
                                  target_agent=self.main_player.env.target_agent,
                                  test_sample_per_agent=test_sample_per_target_agent
                                  )
        print("Mean Reward of %s is %f" % (self.main_player.env.target_agent.config.config_dict['NAME'], reward))
        if reward > best_reward:
            best_reward = reward
            best_player = self.main_player
        for player in self.reference_players:
            reward = self._test_agent(env=player.env.test_env,
                                      target_agent=player.env.target_agent,
                                      test_sample_per_agent=test_sample_per_target_agent
                                      )
            print("Mean Reward of %s is %f" % (player.env.target_agent.config.config_dict['NAME'], reward))

            if reward > best_reward:
                best_reward = reward
                best_player = player

        print("Best player is %s its reward is %f" % (
            best_player.env.target_agent.config.config_dict['NAME'], best_reward))

        best_player.env.target_agent.status = best_player.env.target_agent.status_key['TEST']
        best_reward_log_content = best_player.env.target_agent.log_file_content
        file_name = best_player.env.target_agent.config.config_dict['NAME'] + '_BEST_AGENT_TEST_REWARD.json'

        self.main_player.logger.out_to_file(
            file_path=os.path.join(self.main_player.logger.loss_file_log_dir, file_name),
            content=best_reward_log_content)
        for player in self.reference_players:
            player.logger.out_to_file(
                file_path=os.path.join(player.logger.loss_file_log_dir, file_name),
                content=best_reward_log_content)

    def _test_agent(self, env, target_agent, test_sample_per_agent):
        target_agent.env_status = target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS']
        target_agent.status = target_agent.status_key['TEST']
        test_data = target_agent.sample(env=env,
                                        sample_count=test_sample_per_agent,
                                        store_flag=False,
                                        agent_print_log_flag=True)
        mean_reward = float(np.mean(test_data.reward_set))
        return mean_reward


if __name__ == '__main__':
    pass
