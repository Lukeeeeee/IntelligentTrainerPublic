import os

ENV_BOUND_CONFIG = os.path.dirname(os.path.realpath(__file__))


def get_bound_file(env_name):
    return ENV_BOUND_CONFIG + '/' + env_name + '_state_action_bound.json'
