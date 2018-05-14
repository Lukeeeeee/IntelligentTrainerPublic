#!/usr/bin/env python3

import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from src.env.costDoneFunction import COST_FUNCTION_ENV_DICT, DONE_FUNCTION_ENV_DICT
from src.env.utils import GAME_ENV_NAME_DICT
import util.utilNew as util_new
from config.envBound import get_bound_file
import tensorflow as tf
from src.util.plotter import Plotter
from config.defaultConfig import DEFAULT_CONFIG
from src.core import AssembleGamePlayer


def run_single_experiment(game_env_name, cuda_device, config_set_path, main_intelligent_model_type,
                          main_target_model_type, model_config_dict, referencing_player_config_path_list,
                          referencing_player_model_list, seed=None):
    assemble_player, sess = util_new.create_assemble_game(env_id=game_env_name,
                                                          main_player_config_path=(config_set_path, model_config_dict),
                                                          referencing_player_config_path_list=referencing_player_config_path_list,
                                                          referencing_player_model_list=referencing_player_model_list,
                                                          bound_file=get_bound_file(env_name=game_env_name),
                                                          done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                          cuda_device=cuda_device,
                                                          cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                          main_intelligent_model_type=main_intelligent_model_type,
                                                          main_target_model_type=main_target_model_type)
    try:
        assemble_player.play(seed)
    except KeyboardInterrupt:
        assemble_player.print_log_to_file()
        assemble_player.save_all_model()


def run_multiple_experiments(game_env_name, cuda_device, config_set_path, main_intelligent_model_type,
                             main_target_model_type, model_config_dict, referencing_player_config_path_list,
                             referencing_player_model_list, num, seed=None):
    log_dir_path = []
    if seed is None:
        # seed = [0, 1, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444]
        seed = [55, 66, 77, 88, 99, 111, 222, 333, 444]
    for i in range(num):
        tf.reset_default_graph()
        assemble_player, sess = util_new.create_assemble_game(env_id=game_env_name,
                                                              main_player_config_path=(
                                                                  config_set_path, model_config_dict),
                                                              referencing_player_config_path_list=referencing_player_config_path_list,
                                                              referencing_player_model_list=referencing_player_model_list,
                                                              bound_file=get_bound_file(env_name=game_env_name),
                                                              done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                              cuda_device=cuda_device,
                                                              cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                              main_intelligent_model_type=main_intelligent_model_type,
                                                              main_target_model_type=main_target_model_type)
        log_dir_path.append(assemble_player.main_player.logger.log_dir)
        try:
            assemble_player.play(seed[i])
        except KeyboardInterrupt:
            assemble_player.print_log_to_file()
            assemble_player.save_all_model()

        sess = tf.get_default_session()
        if sess:
            sess.__exit__(None, None, None)
    for log in log_dir_path:
        print(log)
    Plotter.plot_multiply_target_agent_reward(path_list=log_dir_path)


if __name__ == '__main__':
    from config.configSet_HalfCheetah_Intel import CONFIG_SET_HALFCHEETAH_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL
    from config.configSet_HalfCheetah_Intel_No_Dyna import CONFIG_SET_HALFCHEETAH_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_NO_DYNA
    from config.configSet_HalfCheetah_Intel_Random import CONFIG_SET_HALFCHEETAH_INTEL_RANDOM, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_RANDOM

    from config.configSet_MountainCarContinuous_Intel import CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL
    from config.configSet_MountainCarContinuous_Intel_Random import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_RANDOM, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_RANDOM
    from config.configSet_MountainCarContinuous_Intel_No_Dyna import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_NO_DYNA

    from config.configSet_Swimmer_Intel import CONFIG_SET_SWIMMER_INTEL, MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL
    from config.configSet_Swimmer_Intel_Random import CONFIG_SET_SWIMMER_INTEL_RANDOM, \
        MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_RANDOM
    from config.configSet_Swimmer_Intel_No_Dyna import CONFIG_SET_SWIMMER_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_NO_DYNA

    from config.configSet_Reacher_Intel import CONFIG_SET_REACHER_INTEL, MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL
    from config.configSet_Reacher_Intel_Random import CONFIG_SET_REACHER_INTEL_RANDOM, \
        MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_RANDOM
    from config.configSet_Reacher_Intel_No_Dyna import CONFIG_SET_REACHER_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_NO_DYNA

    from config.configSet_Pendulum_Intel import CONFIG_SET_PENDULUM_INTEL, MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL
    from config.configSet_Pendulum_Intel_Random import CONFIG_SET_PENDULUM_INTEL_RANDOM, \
        MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_RANDOM
    from config.configSet_Pendulum_Intel_No_Dyna import CONFIG_SET_PENDULUM_INTEL_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_NO_DYNA

    run_multiple_experiments(game_env_name='HalfCheetah',
                             cuda_device=1,
                             config_set_path=CONFIG_SET_HALFCHEETAH_INTEL,
                             main_intelligent_model_type='DQN',
                             main_target_model_type='TRPO',
                             model_config_dict=MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL['HalfCheetah'],
                             referencing_player_config_path_list=((CONFIG_SET_HALFCHEETAH_INTEL_RANDOM,
                                                                   MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_RANDOM[
                                                                       'HalfCheetah']),
                                                                  (CONFIG_SET_HALFCHEETAH_INTEL_NO_DYNA,
                                                                   MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_NO_DYNA[
                                                                       'HalfCheetah'])),
                             referencing_player_model_list=(('RANDOM', 'TRPO'), ('FIX', 'TRPO')),
                             num=5)

    # run_multiple_experiments(game_env_name='Pendulum-v0',
    #                          cuda_device=2,
    #                          config_set_path=CONFIG_SET_PENDULUM_INTEL,
    #                          main_intelligent_model_type='DQN',
    #                          main_target_model_type='DDPG',
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL['Pendulum-v0'],
    #                          referencing_player_config_path_list=((CONFIG_SET_PENDULUM_INTEL_RANDOM,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_RANDOM[
    #                                                                    'Pendulum-v0']),
    #                                                               (CONFIG_SET_PENDULUM_INTEL_NO_DYNA,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_NO_DYNA[
    #                                                                    'Pendulum-v0'])),
    #                          referencing_player_model_list=(('RANDOM', 'DDPG'), ('FIX', 'DDPG')),
    #                          num=5)
    #
    # run_multiple_experiments(game_env_name='MountainCarContinuous-v0',
    #                          cuda_device=0,
    #                          config_set_path=CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL,
    #                          main_intelligent_model_type='DQN',
    #                          main_target_model_type='DDPG',
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL['MountainCarContinuous-v0'],
    #                          referencing_player_config_path_list=((CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_RANDOM,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_RANDOM[
    #                                                                    'MountainCarContinuous-v0']),
    #                                                               (CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL_NO_DYNA,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL_NO_DYNA[
    #                                                                    'MountainCarContinuous-v0'])),
    #                          referencing_player_model_list=(('RANDOM', 'DDPG'), ('FIX', 'DDPG')),
    #                          num=5)
    #
    # run_multiple_experiments(game_env_name='Reacher-v1',
    #                          cuda_device=1,
    #                          config_set_path=CONFIG_SET_REACHER_INTEL,
    #                          main_intelligent_model_type='DQN',
    #                          main_target_model_type='TRPO',
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL['Reacher-v1'],
    #                          referencing_player_config_path_list=((CONFIG_SET_REACHER_INTEL_RANDOM,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_RANDOM[
    #                                                                    'Reacher-v1']),
    #                                                               (CONFIG_SET_REACHER_INTEL_NO_DYNA,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_NO_DYNA[
    #                                                                    'Reacher-v1'])),
    #                          referencing_player_model_list=(('RANDOM', 'TRPO'), ('FIX', 'TRPO')),
    #                          num=5)

    # run_multiple_experiments(game_env_name='Swimmer-v1',
    #                          cuda_device=0,
    #                          config_set_path=CONFIG_SET_SWIMMER_INTEL,
    #                          main_intelligent_model_type='DQN',
    #                          main_target_model_type='TRPO',
    #                          model_config_dict=MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL['Swimmer-v1'],
    #                          referencing_player_config_path_list=((CONFIG_SET_SWIMMER_INTEL_RANDOM,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_RANDOM[
    #                                                                    'Swimmer-v1']),
    #                                                               (CONFIG_SET_SWIMMER_INTEL_NO_DYNA,
    #                                                                MODEL_NET_WORK_CONFIG_DICT_SWIMMER_INTEL_NO_DYNA[
    #                                                                    'Swimmer-v1'])),
    #                          referencing_player_model_list=(('RANDOM', 'TRPO'), ('FIX', 'TRPO')),
    #                          num=5)
