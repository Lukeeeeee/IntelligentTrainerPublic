from src.agent.agent import Agent
from src.config.config import Config
from conf.key import CONFIG_KEY
from src.agent.baselineTrainerAgent.baselineTrainerAgent import BaselineTrainerAgent
import tensorflow as tf
import numpy as np
import easy_tf_log
import config as cfg


class IntelligentTrainerAgent(Agent):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/intelligentTrainerAgentKey.json')

    def __init__(self, config, model, env):
        super(IntelligentTrainerAgent, self).__init__(config=config,
                                                      model=model,
                                                      env=env)
        self.sample_count = 0
        self.sess = tf.get_default_session()
        self.prev_action = self.model.action_iterator[np.random.randint(len(self.model.action_iterator))]
        self.remain_action_flag = True

    def predict(self, state, *args, **kwargs):
        if self.remain_action_flag is True:
            # self.remain_action_flag = False
            print("self.remain_action_flag is True")
            return (np.array(self.prev_action))*1.0
        else:
            self.remain_action_flag = True
            if self.assigned_action is not None:
                print("self.assigned_action is True")
                ac = list(self.assigned_action)
                self.assigned_action = None
                state = np.reshape(state, [1, -1])
                re = np.array(self.model.predict(self.sess, state))
                if len(re) > len(ac):
                    for i in range(len(ac), len(re)):
                        ac.append(re[i])
                self.sample_count += 1
                if 'F1=0' in cfg.config_dict and cfg.config_dict['F1=0'] is True:
                    ac[0] = 0.0
                if 'F2=0' in cfg.config_dict and cfg.config_dict['F2=0'] is True:
                    ac[1] = 0.0
                return np.array(ac)*1.0
            else:
                print("self.assigned_action is False")
                state = np.reshape(state, [1, -1])
                count = self.sample_count
                eps = 1.0 - (self.config.config_dict['EPS'] - self.config.config_dict['EPS_GREEDY_FINAL_VALUE']) * \
                      (count / self.config.config_dict['EPS_ZERO_FLAG'])
                if eps < 0:
                    eps = 0.0
                rand_eps = np.random.rand(1)
                # print("a_self.model.action_iterator=", self.model._action_iterator)
                if self.config.config_dict['EPS_GREEDY_FLAG'] == 1 and rand_eps < eps:
                    print("using eps greedy random action")
                    res = self.model.action_iterator[np.random.randint(len(self.model.action_iterator))]
                else:
                    print("using action from model")
                    res = np.array(self.model.predict(self.sess, state))

                # print("b_self.model.action_iterator=", self.model.action_iterator)
                if 'F1=0' in cfg.config_dict and cfg.config_dict['F1=0'] is True:
                    res[0] = 0.0
                if 'F2=0' in cfg.config_dict and cfg.config_dict['F2=0'] is True:
                    res[1] = 0.0
                self.sample_count += 1
                return res*1.0

    def update(self):
        # TODO finish your own update by using API with self.model
        self.model.update()

    def store_one_sample(self, state, next_state, action, reward, done, *arg, **kwargs):
        # TODO store the one sample to whatever you want
        if self.model and hasattr(self.model, 'print_log_queue') and callable(self.model.print_log_queue):
            self.model.print_log_queue(status=self.status)
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
