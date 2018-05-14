from src.agent.agent import Agent
from src.config.config import Config
from config.key import CONFIG_KEY
from src.agent.baselineTrainerAgent.baselineTrainerAgent import BaselineTrainerAgent
import tensorflow as tf
import numpy as np
import easy_tf_log


class IntelligentTrainerAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/intelligentTrainerAgentKey.json')

    def __init__(self, config, model, env):
        super(IntelligentTrainerAgent, self).__init__(config=config,
                                                      model=model,
                                                      env=env)
        self.sample_count = 0
        self.sess = tf.get_default_session()

    def predict(self, state, *args, **kwargs):
        sess = tf.get_default_session()
        state = np.reshape(state, [1, -1])
        count = self.sample_count
        eps = 1.0 - (self.config.config_dict['EPS'] - self.config.config_dict['EPS_GREEDY_FINAL_VALUE']) * \
              (count / self.config.config_dict['EPS_ZERO_FLAG'])
        if eps < 0:
            eps = 0.0
        rand_eps = np.random.rand(1)
        if self.config.config_dict['EPS_GREEDY_FLAG'] == 1 and rand_eps < eps:
            res = self.model.action_iterator[np.random.randint(len(self.model.action_iterator))]
        else:
            res = np.array(self.model.predict(self.sess, state))

        for i in range(len(res)):
            easy_tf_log.tflog(key='INTELLIGENT_AGENT_ACTION_DIM_' + str(i), value=res[i])
        self.sample_count += 1
        return res

    def update(self):
        # TODO finish your own update by using API with self.model
        self.model.update()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        # TODO store the one sample to whatever you want

        self.model.store_one_sample(state=state,
                                    next_state=next_state,
                                    action=action,
                                    reward=reward,
                                    done=done)
        self.log_file_content.append({
            'STATE': np.array(state).tolist(),
            'NEW_STATE': np.array(next_state).tolist(),
            'ACTION': np.array(action).tolist(),
            'REWARD': reward,
            'DONE': done,
            'INDEX': self.log_print_count
        })
        self.log_print_count += 1

    def init(self):
        # TODO init your agent and your model
        # this function will be called at the start of the whole train process
        self.model.init()
