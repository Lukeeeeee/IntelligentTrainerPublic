import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from src.env.util import *
import util.utilNew as util_new
from conf.envBound import get_bound_file
import tensorflow as tf
from src.util.plotter import Plotter
from util.utilNew import create_tmp_config_file
import config as cfg
import time


def run_multiple_experiments(game_env_name, cuda_device, player_config_path_list, player_target_model_type_list, num,
                             seed=None, exp_end_with=''):
    log_dir_path = []
    if seed is None:
        seed = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    tmp_path_list = []
    num = num
    cfg.config_dict = {
        'NOT_TRPO_CLEAR_MEMORY': False,
        'STE_V3_TEST_MOVE_OUT': True,
        'NC': 5,
        # 'DISCOUNT': 0.99,
        'SAMPLER_PROB': 0.0,
        'LINEAR_DISCOUNT': 0.7,
        'REF_AGENT': True,
        'COPY_PARTLY': False,
        'F1=0': False,
        'F2=0': False,
        "TIME_SEED": True,
        'BestThre': 0.8,  #####minimum 0.5
        'max_samP': 1.0,
        'MAX_RE_LEN': 30,
        'POW': 2,
        'SPLIT_COUNT': 2,
        'COPY_RATE': 1.0,
        'RANK_REWARD': True,
        'RRThre': 10,
        'WorstRatio': False,
        'SAMPLER_FRE': 3,
        'ZeroThre': 0.95,
    }

    assert int('DISCOUNT' in cfg.config_dict) + int('LINEAR_DISCOUNT' in cfg.config_dict) < 2

    for i in range(len(player_config_path_list)):
        tmp_path = create_tmp_config_file(game_env_name=game_env_name, orgin_config_path=player_config_path_list[i][0])
        tmp_path_list.append((tmp_path, tmp_path))

    for i in range(num):
        if "TIME_SEED" in cfg.config_dict and cfg.config_dict['TIME_SEED'] is True:
            seed[i] = int(round(time.time() * 1000)) % (2 ** 32 - 1)
        tf.reset_default_graph()
        tf.set_random_seed(seed[i])
        assemble_player, sess = util_new.create_random_ensemble_game(env_id=game_env_name,
                                                                     bound_file=get_bound_file(env_name=game_env_name),
                                                                     done_fn=DONE_FUNCTION_ENV_DICT[game_env_name],
                                                                     cuda_device=cuda_device,
                                                                     cost_fn=COST_FUNCTION_ENV_DICT[game_env_name],
                                                                     exp_end_with=exp_end_with,
                                                                     player_config_path_list=tmp_path_list,
                                                                     player_target_model_type_list=player_target_model_type_list,
                                                                     reset_fn=RESET_FUNCTION_ENV_DICT[game_env_name],
                                                                     intelligent_agent_index=0
                                                                     )
        log_dir_path.append(assemble_player.player_list[0].logger.log_dir)
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


if __name__ == '__main__':
    from conf.configSet_Swimmer_Random_Ensemble_Intel import CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_INTEL
    from conf.configSet_Swimmer_Random_Ensemble_Full_Cyber import CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_FULL_CYBER, \
        MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_FULL_CYBER
    from conf.configSet_Swimmer_Random_Ensemble_No_Dyna import CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_NO_DYNA

    from conf.configSet_MountainCarContinuous_Ramdom_Ensemble_Intel import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL, MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL
    from conf.configSet_MountainCarContinuous_Random_Ensemble_Full_Cyber import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_FULL_CYBER, \
        MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_FULL_CYBER
    from conf.configSet_MountainCarContinuous_Random_Ensemble_No_Dyna import \
        CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_NO_CYBER, MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_NO_CYBER

    from conf.configSet_HalfCheetah_Ramdom_Ensemble_Intel import CONFIG_SET_HALFCHEETAH_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL
    from conf.configSet_HalfCheetah_Ramdom_Ensemble_Full_Cyber import CONFIG_SET_HALFCHEETAH_INTEL_ENSEMBLE_FULL_CYBER, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_ENSEMBLE_FULL_CYBER
    from conf.configSet_HalfCheetah_Ramdom_Ensemble_No_Dyna import CONFIG_SET_HALFCHEETAH_INTEL_ENSEMBLE_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_ENSEMBLE_NO_DYNA

    from conf.configSet_Pendulum_Ramdom_Ensemble_Intel import CONFIG_SET_PENDULUM_INTEL, \
        MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL
    from conf.configSet_Pendulum_Ramdom_Ensemble_No_Dyna import CONFIG_SET_PENDULUM_INTEL_ENSEMBLE_NO_DYNA, \
        MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_ENSEMBLE_NO_DYNA
    from conf.configSet_Pendulum_Ramdom_Ensemble_Full_Cyber import CONFIG_SET_PENDULUM_INTEL_FULL_CYBER, \
        MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_ENSEMBLE_FULL_CYBER

    from conf.configSet_Reacher_Random_Ensemble_Intel import CONFIG_SET_REACHER_INTEL_ENSEMBLE, \
        MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_ENSEMBLE
    from conf.configSet_Reacher_Random_Ensemble_No_Dyna import CONFIG_SET_REACHER_NO_DYNA_ENSEMBLE, \
        MODEL_NET_WORK_CONFIG_DICT_REACHER_NO_DYNA_ENSEMBLE
    from conf.configSet_Reacher_Random_Ensemble_Full_Cyber import CONFIG_SET_REACHER_FULL_CYBER_ENSEMBLE, \
        MODEL_NET_WORK_CONFIG_DICT_REACHER_FULL_CYBER_ENSEMBLE
    #
    run_multiple_experiments(game_env_name='Swimmer-v1',
                            cuda_device=-1,
                            player_config_path_list=(
                                (CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_INTEL,
                                 MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_INTEL),
                                (CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_NO_DYNA,
                                 MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_NO_DYNA),
                                (CONFIG_SET_SWIMMER_RANDOM_ENSEMBLE_INTEL,
                                 MODEL_NET_WORK_CONFIG_DICT_SWIMMER_RANDOM_ENSEMBLE_INTEL)),
                            player_target_model_type_list=(('DQN', 'TRPO'), ('FIX', 'TRPO'), ('RANDOM', 'TRPO')),
                            num=1,
                            seed=None,
                            exp_end_with='EnsX4EFinal_FC_BT0.8')
                            #exp_end_with='debug')

    # run_multiple_experiments(game_env_name='Reacher-v1',
    #                           cuda_device=-1,
    #                           player_config_path_list=(
    #                               (CONFIG_SET_REACHER_INTEL_ENSEMBLE,
    #                                MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_ENSEMBLE),
    #                               (CONFIG_SET_REACHER_NO_DYNA_ENSEMBLE,
    #                                MODEL_NET_WORK_CONFIG_DICT_REACHER_NO_DYNA_ENSEMBLE),
    #                               (CONFIG_SET_REACHER_INTEL_ENSEMBLE,
    #                                MODEL_NET_WORK_CONFIG_DICT_REACHER_INTEL_ENSEMBLE)),
    #                           player_target_model_type_list=(('DQN', 'TRPO'), ('FIX', 'TRPO'), ('RANDOM', 'TRPO')),
    #                           num=1,
    #                           seed=None,
    #                           exp_end_with='EnsX4EFinal_FC_BT0.9')
    #                           # exp_end_with='debug')
    # #
    # run_multiple_experiments(game_env_name='Pendulum-v0',
    #                          cuda_device=0,
    #                          player_config_path_list=(
    #                              (CONFIG_SET_PENDULUM_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL),
    #                              (CONFIG_SET_PENDULUM_INTEL_ENSEMBLE_NO_DYNA,
    #                               MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL_ENSEMBLE_NO_DYNA),
    #                              (CONFIG_SET_PENDULUM_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_PENDULUM_INTEL)),
    #                          player_target_model_type_list=(('DQN', 'DDPG'), ('FIX', 'DDPG'), ('RANDOM', 'DDPG')),
    #                          num=1,
    #                          seed=None,
    #                          exp_end_with='EnsX4EFinal_FC_BT0.9')
    #                          # exp_end_with='debug')
    # # #
    # run_multiple_experiments(game_env_name='MountainCarContinuous-v0',
    #                          cuda_device=-1,
    #                          player_config_path_list=(
    #                              (CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL),
    #                              (CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_NO_CYBER,
    #                               MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_NO_CYBER),
    #                              (CONFIG_SET_MOUNTAIN_CAR_CONTINUOUS_CONFIG_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_MOUNTAIN_CAR_CONTINUOUS_INTEL),
    #                          ),
    #                          player_target_model_type_list=(('DQN', 'DDPG'), ('FIX', 'DDPG'), ('RANDOM', 'DDPG')),
    #                          num=1,
    #                          seed=None,
    #                          exp_end_with='EnsX4EFinal_FC_BT0.9')
    #                          # exp_end_with='debug')
    #
    # run_multiple_experiments(game_env_name='HalfCheetah',
    #                          cuda_device=-1,
    #                          player_config_path_list=(
    #                              (CONFIG_SET_HALFCHEETAH_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL),
    #                              (CONFIG_SET_HALFCHEETAH_INTEL_ENSEMBLE_NO_DYNA,
    #                               MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL_ENSEMBLE_NO_DYNA),
    #                              (CONFIG_SET_HALFCHEETAH_INTEL,
    #                               MODEL_NET_WORK_CONFIG_DICT_HALFCHEETAH_INTEL)),
    #                          player_target_model_type_list=(('DQN', 'TRPO'), ('FIX', 'TRPO'), ('RANDOM', 'TRPO')),
    #                          num=1,
    #                          exp_end_with='EnsX4EFinal_FC_BT0.9')
    #                          # exp_end_with='debug')
