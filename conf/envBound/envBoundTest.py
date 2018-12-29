import gym
import numpy as np
import json
from conf.envBound import ENV_BOUND_CONFIG
from src.env.halfCheetahEnv.halfCheetahEnv import HalfCheetahEnvNew


def test_env_bound(env_name=None, env_class=None):
    if env_name:
        env = gym.make(env_name)
    else:
        env = env_class()
        env_name = type(env_class).__name__
    state_list = []
    action_list = []
    state = env.reset()
    for j in range(100):
        print(j)
        env.reset()
        for i in range(1000):
            # print(i)
            state_list.append(state)
            action = env.action_space.sample()
            action_list.append(action)
            state, reward, done, info = env.step(action)
    state_list = np.array(state_list)

    state_low = np.nanmin(state_list, axis=0)
    state_high = np.nanmax(state_list, axis=0)
    action_low = np.nanmin(action_list, axis=0)
    action_high = np.nanmax(action_list, axis=0)

    small_fac = 0.75
    big_fac = 1.5

    for i in range(len(state_low)):
        if state_low[i] < 0.0:
            state_low[i] = state_low[i] * big_fac
        else:
            state_low[i] = state_low[i] * small_fac

    for i in range(len(action_low)):
        if action_low[i] < 0.0:
            action_low[i] = action_low[i] * big_fac
        else:
            action_low[i] = action_low[i] * small_fac

    for i in range(len(state_high)):
        if state_high[i] < 0.0:
            state_high[i] = state_high[i] * small_fac
        else:
            state_high[i] = state_high[i] * big_fac

    for i in range(len(action_high)):
        if action_high[i] < 0.0:
            action_high[i] = action_high[i] * small_fac
        else:
            action_high[i] = action_high[i] * big_fac

    state_low = np.clip(state_low, np.nan_to_num(env.observation_space.low), np.nan_to_num(env.observation_space.high))
    state_high = np.clip(state_high, state_low + 0.001, np.nan_to_num(env.observation_space.high))
    action_low = np.clip(action_low, np.nan_to_num(env.action_space.low), np.nan_to_num(env.action_space.high))
    action_high = np.clip(action_high, action_low + 0.001, np.nan_to_num(env.action_space.high))

    res = {
        "STATE_LOW": list(state_low),
        "STATE_HIGH": list(state_high),
        "ACTION_LOW": list(action_low),
        "ACTION_HIGH": list(action_high),
    }
    with open(file=ENV_BOUND_CONFIG + '/' + env_name + '_state_action_bound.json', mode='w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    test_env_bound(env_name='InvertedPendulum-v1')
    # test_env_bound(env_class=HalfCheetahEnvNew)
