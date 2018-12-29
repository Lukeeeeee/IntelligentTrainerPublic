from src.env.env import BasicEnv
import numpy as np
from src.config.config import Config
from conf.key import CONFIG_KEY
from gym.spaces.box import Box
import src.model.utils.utils as model_util
import copy
import config as cfg


class DynamicsEnv(BasicEnv):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/dynamicsEnvKey.json')

    # TODO Modify cost function

    def __init__(self, config, model, sess, max_episode_steps=1000, init_env=None, cost=None, done=None, reset=None):
        super(DynamicsEnv, self).__init__(config=config)

        self.cost_fn = cost
        self.done_fn = done
        self.reset_fn = reset

        if init_env:
            self.action_space = copy.deepcopy(init_env.action_space)
            self.observation_space = copy.deepcopy(init_env.observation_space)
        else:
            low = np.array(
                [self.config.config_dict['ACTION_LOW'] for i in range(self.config.config_dict['ACTION_SPACE'][0])])
            high = np.array(
                [self.config.config_dict['ACTION_HIGH'] for i in range(self.config.config_dict['ACTION_SPACE'][0])])
            self.action_space = Box(low, high)

            low = np.array(
                [self.config.config_dict['STATE_LOW'] for i in range(self.config.config_dict['STATE_SPACE'][0])])
            high = np.array(
                [self.config.config_dict['STATE_HIGH'] for i in range(self.config.config_dict['STATE_SPACE'][0])])
            self.observation_space = Box(low, high)

        self.model = model
        self.sess = sess
        if 'SWIMMER_HORIZON' in cfg.config_dict:
            self.config.config_dict["MAX_SAMPLE_HORIZON"] = cfg.config_dict['SWIMMER_HORIZON']
        self._max_episode_steps = self.config.config_dict["MAX_SAMPLE_HORIZON"]

        self._max_episode_steps_fixed = self._max_episode_steps
        self._elapsed_steps = 0
        self.state = self.reset()

    @property
    def status(self):
        return self._status

    @property
    def max_step(self):
        return self.config.config_dict["MAX_SAMPLE_HORIZON"]

    def set_max_step(self, val):
        assert 0 < val <= self.config.config_dict["MAX_SAMPLE_HORIZON"]
        self._max_episode_steps = val

    @status.setter
    def status(self, new_value):
        if new_value != self.status_key['TRAIN'] and new_value != self.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)

        if self._status == new_value:
            return
        self._status = new_value
        self.model.status = new_value

    def step(self, action):

        super().step(action=action)
        prev_state = self._get_obs()
        state = self.model.predict(sess=self.sess,
                                   state_input=self._get_obs(),
                                   action_input=action)
        state = model_util.squeeze_array(state, dim=1)

        state = np.clip(state, a_min=self.observation_space.low, a_max=self.observation_space.high)

        reward = self.cost_fn(state=prev_state, action=action, next_state=state)
        info = None

        self._elapsed_steps += 1
        done = self.done_check(prev_state=prev_state, state=state, action=action)

        if done is True:
            state = self.reset()

        self.state = state
        return state, reward, done, info

    def done_check(self, prev_state, state, action):
        if self.done_fn:
            done = self.done_fn(prev_state, state, action)
        else:
            done = False

        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return done

    def fit(self, state_set, action_set, delta_state_label_set, sess):
        loss = self.model.update(sess=sess,
                                 state_input=state_set,
                                 action_input=action_set,
                                 delta_state_label=delta_state_label_set)
        return loss

    def test(self, state_set, action_set, delta_state_label_set, sess):
        self.model.test(sess=sess,
                        state_input=state_set,
                        action_input=action_set,
                        delta_state_label=delta_state_label_set)

    def print_log_queue(self, status):
        self.status = status
        self.model.print_log_queue(status)

    def _get_obs(self):
        return model_util.squeeze_array(np.array(self.state), dim=1)

    def get_state(self, env):
        return env._get_obs()

    def set_state(self, state):
        self.state = model_util.squeeze_array(state, dim=1)

    def close(self):
        print("Close Dynamics Env")

    def _configure(self):
        pass

    def seed(self, seed=None):
        pass

    def reset(self):
        # TODO MODIFY THE RANGE OF
        super().reset()
        self._elapsed_steps = 0
        if self.reset_fn:
            self.state = self.reset_fn()
        else:
            self.state = self.observation_space.sample()
        return self.state

    def init(self):
        super().init()
        if hasattr(self.model, 'init') and callable(self.model.init):
            self.model.init()


if __name__ == '__main__':
    from conf import CONFIG
    from src.model.dynamicsEnvMlpModel.dynamicsEnvMlpModel import DynamicsEnvMlpModel
    import tensorflow as tf

    conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
    conf.load_config(path=CONFIG + '/dynamicsEnvMlpModelTestConfig.json')
    a = DynamicsEnvMlpModel(config=conf)

    conf = Config(standard_key_list=DynamicsEnv.key_list)
    conf.load_config(path=CONFIG + '/dynamicsEnvTestConfig.json')
    sess = tf.Session()
    with sess.as_default():
        a = DynamicsEnv(config=conf, model=a, sess=sess)
        a.init()
        obs = a.reset()
        action = np.zeros([1] + list(a.action_space.shape))
        new_state, reward, _, _ = a.step(action)

        label = a.observation_space.sample()
        a.fit(state_set=obs, action_set=action, delta_state_label_set=label - obs, sess=sess)
