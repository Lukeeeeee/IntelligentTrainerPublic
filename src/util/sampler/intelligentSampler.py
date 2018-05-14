from src.core import Basic
import numpy as np
from src.model.simpleMlpModel.simpleMlpModel import SimpleMlpModel
from src.config.config import Config
from config.key import CONFIG_KEY
import tensorflow as tf
from src.util.sampler.sampler import Sampler


class IntelligentSampler(Sampler):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/intelligentSamplerKey.json')

    def __init__(self, cost_fn, config):
        super().__init__(cost_fn, config)
        MLP_key_list = Config.load_json(file_path=CONFIG_KEY + '/simpleMlpModelKey.json')

        state_estimate_model_conf = Config(standard_key_list=MLP_key_list,
                                           config_dict=config.config_dict['STATE_ESTIMATE_MODEL'])
        state_estimate_model_conf.config_dict['NAME'] = self.config.config_dict['NAME'] + '_STATE_ESTIMATE_MODEL'
        print("state_estimate_model_conf=", state_estimate_model_conf.config_dict)

        self.state_estimate_model = SimpleMlpModel(config=state_estimate_model_conf)
        self.state_estimate_model.init()

        dynamics_error_estimate_model_conf = Config(standard_key_list=MLP_key_list, config_dict=config.config_dict[
            'DYNAMICS_ERROR_ESTIMATE_MODEL'])
        dynamics_error_estimate_model_conf.config_dict['NAME'] = self.config.config_dict[
                                                                     'NAME'] + '_DYNAMICS_ERROR_ESTIMATE_MODEL'
        print("dynamics_error_estimate_model_conf=", dynamics_error_estimate_model_conf)
        self.dynamics_error_estimate_model = SimpleMlpModel(config=dynamics_error_estimate_model_conf)
        self.dynamics_error_estimate_model.init()
        self.F1 = 0.5
        self.F2 = 0.5
        self.count_new_real_samples = 0

    def sample(self, env, agent, sample_count, store_flag=False, agent_print_log_flag=False, reset_Flag=True):
        path = super().sample(env=env,
                              agent=agent,
                              sample_count=sample_count,
                              store_flag=store_flag,
                              agent_print_log_flag=agent_print_log_flag,
                              reset_Flag=reset_Flag)
        if agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            self.count_new_real_samples = sample_count
        return path

    def reset(self, env, agent, reset_Noise=True):
        state = super().reset(env=env, agent=agent, reset_Noise=reset_Noise)
        sess = tf.get_default_session()
        vs_list = []
        if agent.sampler.env_status == agent.sampler.config.config_dict['TEST_ENVIRONMENT_STATUS']:
            return env.reset()
        print("Entered intel reset")

        if agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            pass
            for kkk_ in range(50):
                state = env.reset()
                squo = agent.model.q_value(state=state, step=0)
                squo = self.F1 * squo + (1 - self.F1) * np.random.rand()
                vs_list.append(squo)
                if len(vs_list) > 5 and (np.max(vs_list) - np.min(vs_list)) * 0.999 + np.min(vs_list) < vs_list[-1]:
                    break
        else:
            if np.random.rand() < self.F1:
                idx = np.random.randint(agent.model.real_data_memory.observations0.length)
                state = agent.model.real_data_memory.observations0.get_batch(idx)
                env.set_state(state)
            else:
                state = env.reset()

        return state

    def train(self, state_est_input, state_est_label, dyn_error_est_input, dyn_error_est_label, *args, **kwargs):
        sess = tf.get_default_session()
        for i in range(self.config.config_dict['STEP']):
            # self.state_estimate_model.update(sess=sess,
            #                                  input=state_est_input,
            #                                  label=state_est_label)
            self.dynamics_error_estimate_model.update(sess=sess,
                                                      input=dyn_error_est_input,
                                                      label=dyn_error_est_label)

    def set_F(self, F1, F2):
        self.F1 = F1
        self.F2 = F2

    def print_log_queue(self, status):
        self.status = status
        self.state_estimate_model.print_log_queue(status=status)
        self.dynamics_error_estimate_model.print_log_queue(status=status)
