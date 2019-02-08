from src.agent.agent import Agent
from src.config.config import Config
from conf.key import CONFIG_KEY
import numpy as np
from src.util.sampler.sampler import Sampler
import easy_tf_log
from src.core import Basic
from src.util.noiseAdder import noise_adder
import queue


class TargetAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/ddpgAgentKey.json')

    def __init__(self, config, real_env, cyber_env, model, sampler=Sampler()):
        super(TargetAgent, self).__init__(config=config,
                                          env=real_env,
                                          model=model,
                                          sampler=sampler)
        self.real_env = real_env
        self.cyber_env = cyber_env
        self.env = None
        self._env_status = self.config.config_dict['REAL_ENVIRONMENT_STATUS']
        self._real_env_sample_count = 0
        self._cyber_env_sample_count = 0
        self.SamplerTraingCount = 0
        self._train_real_log_queue = queue.Queue(maxsize=int(1e10))
        self._train_cyber_log_queue = queue.Queue(maxsize=int(1e10))
        self._test_cyber_log_queue = queue.Queue(maxsize=int(1e10))
        self._test_real_log_queue = queue.Queue(maxsize=int(1e10))

    @property
    def env_sample_count(self):
        if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return self._real_env_sample_count
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return self._cyber_env_sample_count

    @env_sample_count.setter
    def env_sample_count(self, new_value):
        assert isinstance(new_value, int) and new_value >= 0
        if self.status == self.status_key['TEST']:
            return
        if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self._real_env_sample_count = new_value
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self._cyber_env_sample_count = new_value

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_value):
        if new_value != Basic.status_key['TRAIN'] and new_value != Basic.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)
        if new_value == Basic.status_key['TEST'] and self.env_status == self.config.config_dict[
            'REAL_ENVIRONMENT_STATUS']:
            self.sampler.env_status = self.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']
        self._status = new_value
        self.model.status = new_value

    @property
    def env_status(self):
        return self._env_status

    @env_status.setter
    def env_status(self, new_sta):
        assert (new_sta == self.config.config_dict['REAL_ENVIRONMENT_STATUS'] or new_sta == self.config.config_dict[
            'CYBER_ENVIRONMENT_STATUS'])
        self._env_status = new_sta
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.env = self.real_env
            self.model.env_status = self.model.config.config_dict['REAL_ENVIRONMENT_STATUS']
            if self.status == self.status_key['TEST']:
                self.sampler.env_status = self.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']
            else:
                self.sampler.env_status = self.sampler.config.config_dict['REAL_ENVIRONMENT_STATUS']
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            self.env = self.cyber_env
            self.model.env_status = self.model.config.config_dict['CYBER_ENVIRONMENT_STATUS']
            self.sampler.env_status = self.sampler.config.config_dict['CYBER_ENVIRONMENT_STATUS']

        else:
            raise ValueError('Wrong Agent Environment Env Status: %d' % new_sta)

    @property
    def current_env_status(self):
        if self._env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            return 'REAL_ENVIRONMENT_STATUS'
        elif self._env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
            return 'CYBER_ENVIRONMENT_STATUS'

    @property
    def log_queue(self):
        if self._status == Basic.status_key['TRAIN']:
            if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
                return self._train_real_log_queue
            elif self.env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
                return self._train_cyber_log_queue
        elif self._status == Basic.status_key['TEST']:
            if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
                return self._test_real_log_queue
            elif self.env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
                return self._test_cyber_log_queue
        raise KeyError('Current Status: %d or Env status: %d did not existed' % (self._status, self._env_status))

    def predict(self, state, step_count=None, *args, **kwargs):
        state = np.reshape(state, [-1])
        count = self._real_env_sample_count
        eps = 1.0 - (self.config.config_dict['EPS'] - self.config.config_dict['EPS_GREEDY_FINAL_VALUE']) * \
              (count / self.config.config_dict['EPS_ZERO_FLAG'])
        if eps < 0:
            eps = 0.0
        rand_eps = np.random.rand(1)
        if self.config.config_dict['EPS_GREEDY_FLAG'] == 1 and rand_eps < eps and self.status == self.status_key[
            'TRAIN']:
            res = self.env.action_space.sample()
        else:
            if step_count is not None:
                res = np.array(self.model.predict(state, step_count=step_count))
            else:
                res = np.array(self.model.predict(state))

        if self.config.config_dict['NOISE_FLAG'] > 0 and self.status == self.status_key['TRAIN']:
            res, noise = noise_adder(action=res, agent=self)
        return np.reshape(res, [-1])

    def sample(self, env, sample_count, store_flag=False, agent_print_log_flag=False, resetNoise_Flag=False):
        if self.status == self.status_key['TEST']:
            self.sampler.reset(env=env, agent=self)
        if self.model.config.config_dict['NOISE_FLAG'] == 2:
            resetNoise_Flag = True
        else:
            resetNoise_Flag = False
        return super().sample(env, sample_count, store_flag, agent_print_log_flag, resetNoise_Flag)

    def train(self, sampler_train_flag=0):

        if self.model.memory_length >= self.model.config.config_dict['BATCH_SIZE']:
            res_dict = self.model.update()
        else:
            res_dict = None

        return res_dict

    def store_one_sample(self, state, next_state, action, reward, done):
        self.model.store_one_sample(state=state,
                                    next_state=next_state,
                                    action=action,
                                    reward=reward,
                                    done=done)

    def return_most_recent_sample(self, sample_count, memory):
        pass

    def init(self):
        self.model.init()
        self.model.reset()
        super().init()

    def print_log_queue(self, status):
        self.status = status
        reward_list = []

        while self.log_queue.qsize() > 0:
            reward_list.append(self.log_queue.get()[self.name + '_SAMPLE_REWARD'])
        print("%s status reward list len is %d" % (self.current_env_status, len(reward_list)))
        if len(reward_list) > 0:
            reward_list = np.array(reward_list)
            sum = np.sum(reward_list).item()
            mean = np.mean(reward_list).item()
            std = np.std(reward_list).item()

            env_status = None
            if self.env_status == self.config.config_dict['REAL_ENVIRONMENT_STATUS']:
                env_status = 'REAL_ENV'
            elif self.env_status == self.config.config_dict['CYBER_ENVIRONMENT_STATUS']:
                env_status = 'CYBER_ENV'
            print("%s %s Reward: Sum: %f Average %f Std %f" % (self.name, env_status, sum, mean, std))

            if self.status == self.status_key['TRAIN']:
                self.log_file_content.append({'INDEX': self.log_print_count,
                                              'REWARD_SUM': sum,
                                              'REWARD_MEAN': mean,
                                              'REWARD_STD': std,
                                              'REAL_SAMPLE_COUNT': self._real_env_sample_count,
                                              'CYBER_SAMPLE_COUNT': self._cyber_env_sample_count,
                                              'ENV': env_status})
            elif self.status == self.status_key['TEST']:
                self.log_file_content.append({'INDEX': self.log_print_count,
                                              'REWARD_SUM': sum,
                                              'REWARD_MEAN': mean,
                                              'REWARD_STD': std,
                                              'REAL_SAMPLE_COUNT': self._real_env_sample_count,
                                              'CYBER_SAMPLE_COUNT': self._cyber_env_sample_count,
                                              'ENV': env_status})
            self.log_print_count += 1
        # TODO HOW TO ELEGANT CHANGE THIS
        if self.model and hasattr(self.model, 'print_log_queue') and callable(self.model.print_log_queue):
            self.model.print_log_queue(status=status)

    def reset(self):
        super().reset()
        self.model.reset()

    def get_trpo_step_count(self):
        from src.model.trpoModel.trpoModel import TrpoModel
        if isinstance(self.model, TrpoModel) is True:
            return self.model.step_count
        else:
            return None


if __name__ == '__main__':
    from conf import CONFIG
    from src.model.ddpgModel.ddpgModel import DDPGModel

    conf = Config(standard_key_list=TargetAgent.key_list)
    conf.load_config(path=CONFIG + '/ddpgAgentTestConfig.json')

    ddog_con = Config(standard_key_list=DDPGModel.key_list)
    ddog_con.load_config(path=CONFIG + '/targetModelTestConfig.json')

    ddpg = DDPGModel(config=ddog_con)

    a = TargetAgent(config=conf,
                    real_env=2,
                    cyber_env=1,
                    model=ddpg)
