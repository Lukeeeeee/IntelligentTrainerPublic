import numpy as np
import config as cfg
from copy import deepcopy as dp

def noise_adder(action, agent):
    noise = []
    env_count = agent._real_env_sample_count

    if agent.config.config_dict['NOISE_FLAG'] == 2 and env_count < agent.config.config_dict[
        'MAX_SAMPLE_COUNT'] * 0.5:
        # if not 'GLOBAL_NOISE' in cfg.config_dict:
        #     cfg.config_dict['GLOBAL_NOISE'] = dp(agent.model.action_noise)
        p = (env_count // 1000) / (agent.config.config_dict['MAX_SAMPLE_COUNT'] * 0.5 / 1000.0)
        noise = (1 - p) * agent.model.action_noise()
        # noise = (1 - p) * cfg.config_dict['GLOBAL_NOISE']()
        action = p * action + noise * 3.0

    elif agent.config.config_dict['NOISE_FLAG'] == 1:
        ep = agent._real_env_sample_count / agent.config.config_dict['MAX_SAMPLE_COUNT'] * \
             agent.config.config_dict['EP_MAX']
        noise_scale = (agent.config.config_dict['INIT_NOISE_SCALE'] * agent.config.config_dict['NOISE_DECAY'] ** ep) * \
                      (agent.real_env.action_space.high - agent.real_env.action_space.low)
        noise = noise_scale * agent.model.action_noise()
        action = action + noise
    elif agent.config.config_dict['NOISE_FLAG'] == 3:
        noise = agent.model.action_noise()
        action = action + noise
    noise = np.reshape(np.array([noise]), [-1]).tolist()

    # action = np.clip(action, [-1], [1])
    # print("action", action)
    action = np.clip(action, [-1], [1])
    return action, noise