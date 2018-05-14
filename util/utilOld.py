import os
import os
import sys
import tensorflow as tf
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')

from src.config.config import Config
from src.model.ddpgModel.ddpgModel import DDPGModel
from src.model.dqnModel.dqnModel import DQNModel
from src.model.trpoModel.trpoModel import TrpoModel
from src.agent.targetAgent.targetAgent import TargetAgent
from src.env.dynamicsEnv.dynamicsEnv import DynamicsEnv
from src.env.trainerEnv.trainerEnv import TrainerEnv
from src.model.fxiedOutputModel.FixedOutputModel import FixedOutputModel
from src.agent.baselineTrainerAgent.baselineTrainerAgent import BaselineTrainerAgent
from src.agent.IntelligentTrainerAgent.intelligentRandomTrainerAgent import IntelligentRandomTrainerAgent
from src.model.dynamicsEnvMlpModel.dynamicsEnvMlpModel import DynamicsEnvMlpModel
from src.model.reinforceModel.reinforceModel import REINFORCEModel
from src.core import GamePlayer
from src.agent.IntelligentTrainerAgent.intelligentTrainerAgent import IntelligentTrainerAgent
from src.util.sampler.sampler import Sampler
from src.env.trainerEnv.baselineTrainerEnv import BaselineTrainerEnv
from src.env.utils import make_env
from src.util.sampler.intelligentSampler import IntelligentSampler
from src.util.sampler.fakeSampler import FakeSampler
from config.defaultConfig import DEFAULT_CONFIG


def create_baseline_game_with_trpo(env_id, game_specific_config_path, bound_file, cost_fn=None, cuda_device=0,
                                   done_fn=None, config_set_path=DEFAULT_CONFIG):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        trpo_config = Config(standard_key_list=TrpoModel.key_list)
        trpo_config.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=BaselineTrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/baselineTrainerEnvTestConfig.json')

        baseline_trainer_model_config = Config(standard_key_list=FixedOutputModel.key_list)
        baseline_trainer_model_config.load_config(path=config_set_path + '/baselineTrainerModelTestConfig.json')
        print("baseline_trainer_model_config=", baseline_trainer_model_config)

        baseline_trainer_agent_config = Config(standard_key_list=BaselineTrainerAgent.key_list)
        baseline_trainer_agent_config.load_config(path=config_set_path + '/baselineTrainerAgentTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=trpo_config)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf,
                               sess=sess,
                               model=dynamics_model,
                               init_env=real_env,
                               cost=cost_fn,
                               done=done_fn)

        trpo = TrpoModel(config=trpo_config, action_bound=action_bound, obs_bound=obs_bound)

        a = TargetAgent(config=target_agent_config,
                        real_env=real_env,
                        cyber_env=dyna_env,
                        model=trpo,
                        sampler=Sampler(cost_fn=cost_fn))

        env = BaselineTrainerEnv(config=trainer_env_conf,
                                 cyber_env=dyna_env,
                                 real_env=real_env,
                                 target_agent=a,
                                 test_env=test_env)

        baseline_trainer_model = FixedOutputModel(config=baseline_trainer_model_config)

        trainer_agent = BaselineTrainerAgent(config=baseline_trainer_agent_config,
                                             model=baseline_trainer_model,
                                             env=env)
        basic_list = [dynamics_model, dyna_env, trpo, a, env, baseline_trainer_model, trainer_agent]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=0)
        return player, sess


def create_baseline_game(env_id, game_specific_config_path, bound_file, cost_fn=None, cuda_device=0,
                         done_fn=None, config_set_path=DEFAULT_CONFIG):
    # do some basic test
    # Create a DDPG model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        real_env = make_env(env_id)

        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        ddpg_conf = Config(standard_key_list=DDPGModel.key_list)
        ddpg_conf.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=BaselineTrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/baselineTrainerEnvTestConfig.json')

        baseline_trainer_model_config = Config(standard_key_list=FixedOutputModel.key_list)
        baseline_trainer_model_config.load_config(path=config_set_path + '/baselineTrainerModelTestConfig.json')
        print("baseline_trainer_model_config=", baseline_trainer_model_config)

        baseline_trainer_agent_config = Config(standard_key_list=BaselineTrainerAgent.key_list)
        baseline_trainer_agent_config.load_config(path=config_set_path + '/baselineTrainerAgentTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=ddpg_conf)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf,
                               sess=sess,
                               model=dynamics_model,
                               init_env=real_env,
                               cost=cost_fn,
                               done=done_fn)

        ddpg = DDPGModel(config=ddpg_conf, action_bound=action_bound, obs_bound=obs_bound)

        a = TargetAgent(config=target_agent_config,
                        real_env=real_env,
                        cyber_env=dyna_env,
                        model=ddpg,
                        sampler=Sampler(cost_fn=cost_fn))

        env = BaselineTrainerEnv(config=trainer_env_conf,
                                 cyber_env=dyna_env,
                                 real_env=real_env,
                                 target_agent=a,
                                 test_env=test_env)

        baseline_trainer_model = FixedOutputModel(config=baseline_trainer_model_config)

        trainer_agent = BaselineTrainerAgent(config=baseline_trainer_agent_config,
                                             model=baseline_trainer_model,
                                             env=env)
        basic_list = [dynamics_model, dyna_env, ddpg, a, env, baseline_trainer_model, trainer_agent]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=0)
        return player, sess


def create_intelligent_game(env_id, game_specific_config_path, bound_file, cost_fn=None, cuda_device=0,
                            done_fn=None, config_set_path=DEFAULT_CONFIG):
    # Create a DDPG model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        # Create a real env
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        # Create config for every model
        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        ddpg_conf = Config(standard_key_list=DDPGModel.key_list)
        ddpg_conf.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')
        ddpg_conf.config_dict['NAME'] = 'DDPG'

        trainer_env_conf = Config(standard_key_list=TrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/trainerEnvTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=ddpg_conf)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dynamics_model, init_env=real_env,
                               cost=cost_fn, done=done_fn)

        ddpg = DDPGModel(config=ddpg_conf, action_bound=action_bound, obs_bound=obs_bound)

        sampler_conf = Config(standard_key_list=IntelligentSampler.key_list)
        sampler_conf.load_config(path=config_set_path + '/intelligentSamplerConfig.json')

        sampler = IntelligentSampler(cost_fn=cost_fn, config=sampler_conf)

        ddpg_agent = TargetAgent(config=target_agent_config,
                                 real_env=real_env,
                                 cyber_env=dyna_env,
                                 model=ddpg,
                                 sampler=sampler)

        env = TrainerEnv(config=trainer_env_conf,
                         cyber_env=dyna_env,
                         real_env=real_env,
                         target_agent=ddpg_agent,
                         test_env=test_env)

        intelligent_trainer_model_config = Config(standard_key_list=DQNModel.key_list)
        intelligent_trainer_model_config.load_config(path=config_set_path + '/intelligentTrainerModelTestConfig.json')

        intelligent_trainer_agent_config = Config(standard_key_list=IntelligentTrainerAgent.key_list)
        intelligent_trainer_agent_config.load_config(path=config_set_path + '/intelligentTrainerAgentTestConfig.json')

        intelligent_trainer_model = DQNModel(config=intelligent_trainer_model_config,
                                             action_bound=(env.action_space.low, env.action_space.high))

        trainer_agent = IntelligentTrainerAgent(config=intelligent_trainer_agent_config,
                                                model=intelligent_trainer_model,
                                                env=env)
        # END

        basic_list = [dynamics_model, dyna_env, ddpg, ddpg_agent, env, intelligent_trainer_model, trainer_agent,
                      sampler]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=1)
        return player, sess


def create_intelligent_game_with_trpo(env_id, game_specific_config_path, bound_file, cost_fn=None, cuda_device=0,
                                      done_fn=None, config_set_path=DEFAULT_CONFIG):
    # Create a DDPG model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        # Create a real env
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        # Create config for every model
        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        trpo_config = Config(standard_key_list=TrpoModel.key_list)
        trpo_config.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=TrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/trainerEnvTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=trpo_config)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dynamics_model, init_env=real_env,
                               cost=cost_fn, done=done_fn)

        trpo = TrpoModel(config=trpo_config, action_bound=action_bound, obs_bound=obs_bound)

        sampler_conf = Config(standard_key_list=IntelligentSampler.key_list)
        sampler_conf.load_config(path=config_set_path + '/intelligentSamplerConfig.json')

        sampler = IntelligentSampler(cost_fn=cost_fn, config=sampler_conf)

        ddpg_agent = TargetAgent(config=target_agent_config,
                                 real_env=real_env,
                                 cyber_env=dyna_env,
                                 model=trpo,
                                 sampler=sampler)

        env = TrainerEnv(config=trainer_env_conf,
                         cyber_env=dyna_env,
                         real_env=real_env,
                         target_agent=ddpg_agent,
                         test_env=test_env)

        intelligent_trainer_model_config = Config(standard_key_list=DQNModel.key_list)
        intelligent_trainer_model_config.load_config(path=config_set_path + '/intelligentTrainerModelTestConfig.json')

        intelligent_trainer_agent_config = Config(standard_key_list=IntelligentTrainerAgent.key_list)
        intelligent_trainer_agent_config.load_config(path=config_set_path + '/intelligentTrainerAgentTestConfig.json')

        intelligent_trainer_model = DQNModel(config=intelligent_trainer_model_config,
                                             action_bound=(env.action_space.low, env.action_space.high))

        trainer_agent = IntelligentTrainerAgent(config=intelligent_trainer_agent_config,
                                                model=intelligent_trainer_model,
                                                env=env)
        # END

        basic_list = [dynamics_model, dyna_env, trpo, ddpg_agent, env, intelligent_trainer_model, trainer_agent]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=1)
        return player, sess


def create_intelligent_game_with_reinforce(env_id, game_specific_config_path, bound_file, cost_fn=None, cuda_device=0,
                                           done_fn=None, config_set_path=DEFAULT_CONFIG):
    # Create a DDPG model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        # Create a real env
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        # Create config for every model
        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        ddpg_conf = Config(standard_key_list=DDPGModel.key_list)
        ddpg_conf.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=TrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/trainerEnvTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=ddpg_conf)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dynamics_model, init_env=real_env,
                               cost=cost_fn, done=done_fn)

        ddpg = DDPGModel(config=ddpg_conf, action_bound=action_bound, obs_bound=obs_bound)

        sampler_conf = Config(standard_key_list=IntelligentSampler.key_list)
        sampler_conf.load_config(path=config_set_path + '/intelligentSamplerConfig.json')

        sampler = IntelligentSampler(cost_fn=cost_fn, config=sampler_conf)

        ddpg_agent = TargetAgent(config=target_agent_config,
                                 real_env=real_env,
                                 cyber_env=dyna_env,
                                 model=ddpg,
                                 sampler=sampler)

        env = TrainerEnv(config=trainer_env_conf,
                         cyber_env=dyna_env,
                         real_env=real_env,
                         target_agent=ddpg_agent,
                         test_env=test_env)

        intelligent_trainer_model_config = Config(standard_key_list=REINFORCEModel.key_list)
        intelligent_trainer_model_config.load_config(path=config_set_path + '/intelligentTrainerModelTestConfig.json')

        intelligent_trainer_agent_config = Config(standard_key_list=IntelligentTrainerAgent.key_list)
        intelligent_trainer_agent_config.load_config(path=config_set_path + '/intelligentTrainerAgentTestConfig.json')

        # intelligent_trainer_model_config.config_dict['LEARNING_RATE'] = 0.01
        intelligent_trainer_model = REINFORCEModel(config=intelligent_trainer_model_config,
                                                   action_bound=(env.action_space.low, env.action_space.high))

        trainer_agent = IntelligentTrainerAgent(config=intelligent_trainer_agent_config,
                                                model=intelligent_trainer_model,
                                                env=env)
        # END

        basic_list = [dynamics_model, dyna_env, ddpg, ddpg_agent, env, intelligent_trainer_model, trainer_agent,
                      sampler]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=1)
        return player, sess


def create_intelligent_game_with_trpo_reinforce(env_id, game_specific_config_path, bound_file, cost_fn=None,
                                                cuda_device=0,
                                                done_fn=None, config_set_path=DEFAULT_CONFIG):
    # Create a DDPG model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    with sess.as_default():
        # Create a real env
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        # Create config for every model
        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=config_set_path + '/ddpgAgentTestConfig.json')

        trpo_config = Config(standard_key_list=TrpoModel.key_list)
        trpo_config.load_config(path=game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=TrainerEnv.key_list)
        trainer_env_conf.load_config(path=config_set_path + '/trainerEnvTestConfig.json')

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=config_set_path + '/gamePlayerTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=trpo_config)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dynamics_model, init_env=real_env,
                               cost=cost_fn, done=done_fn)

        trpo = TrpoModel(config=trpo_config, action_bound=action_bound, obs_bound=obs_bound)

        sampler_conf = Config(standard_key_list=IntelligentSampler.key_list)
        sampler_conf.load_config(path=config_set_path + '/intelligentSamplerConfig.json')

        sampler = IntelligentSampler(cost_fn=cost_fn, config=sampler_conf)

        ddpg_agent = TargetAgent(config=target_agent_config,
                                 real_env=real_env,
                                 cyber_env=dyna_env,
                                 model=trpo,
                                 sampler=sampler)

        env = TrainerEnv(config=trainer_env_conf,
                         cyber_env=dyna_env,
                         real_env=real_env,
                         target_agent=ddpg_agent,
                         test_env=test_env)

        intelligent_trainer_model_config = Config(standard_key_list=REINFORCEModel.key_list)
        intelligent_trainer_model_config.load_config(path=config_set_path + '/intelligentTrainerModelTestConfig.json')

        intelligent_trainer_agent_config = Config(standard_key_list=IntelligentTrainerAgent.key_list)
        intelligent_trainer_agent_config.load_config(path=config_set_path + '/intelligentTrainerAgentTestConfig.json')

        intelligent_trainer_model = REINFORCEModel(config=intelligent_trainer_model_config,
                                                   action_bound=(env.action_space.low, env.action_space.high))

        trainer_agent = IntelligentTrainerAgent(config=intelligent_trainer_agent_config,
                                                model=intelligent_trainer_model,
                                                env=env)
        # END

        basic_list = [dynamics_model, dyna_env, trpo, ddpg_agent, env, intelligent_trainer_model, trainer_agent]

        player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list, ep_type=1)
        return player, sess


def create_assemble_game(env_id, assemble_game_specific_config_path,
                         game_specific_config_path,
                         assemble_config_set_path, config_set_path, bound_file,
                         cost_fn=None, cuda_device=0,
                         done_fn=None):
    player, sess = create_baseline_game_with_trpo(env_id=env_id,
                                                  game_specific_config_path=game_specific_config_path,
                                                  bound_file=bound_file,
                                                  cost_fn=cost_fn,
                                                  cuda_device=cuda_device,
                                                  done_fn=done_fn,
                                                  config_set_path=config_set_path)
    assemble_player_1 = create_assemble_intelligent_game_with_trpo_reinforce(env_id=env_id,
                                                                             bound_file=bound_file,
                                                                             cost_fn=cost_fn,
                                                                             done_fn=done_fn,
                                                                             player=player,
                                                                             sess=sess,
                                                                             assemble_config_set_path=assemble_config_set_path,
                                                                             assemble_game_specific_config_path=assemble_game_specific_config_path,
                                                                             name='_ASSEMBLE_1')
    assemble_player_2 = create_assemble_intelligent_game_with_trpo_reinforce(env_id=env_id,
                                                                             bound_file=bound_file,
                                                                             cost_fn=cost_fn,
                                                                             done_fn=done_fn,
                                                                             player=player,
                                                                             sess=sess,
                                                                             assemble_config_set_path=assemble_config_set_path,
                                                                             assemble_game_specific_config_path=assemble_game_specific_config_path,
                                                                             name='_ASSEMBLE_2')
    return player, [assemble_player_1, assemble_player_2], sess


def create_assemble_intelligent_game_with_trpo_reinforce(env_id, player, sess, assemble_game_specific_config_path,
                                                         assemble_config_set_path, bound_file,
                                                         cost_fn=None,
                                                         done_fn=None,
                                                         name='_ASSEMBLE'):
    with sess.as_default():
        real_env = make_env(env_id)
        test_env = make_env(env_id)

        bound = Config.load_json(file_path=bound_file)

        action_bound = (np.array(bound['ACTION_LOW']), np.array(bound['ACTION_HIGH']))
        obs_bound = (np.array(bound['STATE_LOW']), np.array(bound['STATE_HIGH']))

        # Create config for every model

        dynamics_model_conf = Config(standard_key_list=DynamicsEnvMlpModel.key_list)
        dynamics_model_conf.load_config(path=assemble_game_specific_config_path + '/dynamicsEnvMlpModelTestConfig.json')

        dynamics_env_conf = Config(standard_key_list=DynamicsEnv.key_list)
        dynamics_env_conf.load_config(path=assemble_config_set_path + '/dynamicsEnvTestConfig.json')

        target_agent_config = Config(standard_key_list=TargetAgent.key_list)
        target_agent_config.load_config(path=assemble_config_set_path + '/ddpgAgentTestConfig.json')

        trpo_config = Config(standard_key_list=TrpoModel.key_list)
        trpo_config.load_config(path=assemble_game_specific_config_path + '/targetModelTestConfig.json')

        trainer_env_conf = Config(standard_key_list=TrainerEnv.key_list)
        trainer_env_conf.load_config(path=assemble_config_set_path + '/trainerEnvTestConfig.json')

        trainer_env_conf.config_dict['DYNAMICS_TRAIN_ITERATION'] = 0
        # trainer_env_conf.config_dict['TRAIN_DYNAMICS_AFTER_SAMPLE_COUNT'] = 100000000000

        player_config = Config(standard_key_list=GamePlayer.key_list)
        player_config.load_config(path=assemble_config_set_path + '/gamePlayerTestConfig.json')

        baseline_trainer_model_config = Config(standard_key_list=FixedOutputModel.key_list)
        baseline_trainer_model_config.load_config(
            path=assemble_config_set_path + '/baselineTrainerModelTestConfig.json')

        baseline_trainer_agent_config = Config(standard_key_list=BaselineTrainerAgent.key_list)
        baseline_trainer_agent_config.load_config(
            path=assemble_config_set_path + '/baselineTrainerAgentTestConfig.json')

        player_config.config_dict['GAME_NAME'] = env_id

        update_state_action_space_by_env(real_env=real_env,
                                         dynamics_model_conf=dynamics_model_conf,
                                         dynamics_env_conf=dynamics_env_conf,
                                         ddpg_conf=trpo_config)

        update_name(config_list=[dynamics_model_conf, target_agent_config, trpo_config, trainer_env_conf, player_config,
                                 baseline_trainer_model_config, baseline_trainer_agent_config],
                    name=name)

        dynamics_model = DynamicsEnvMlpModel(config=dynamics_model_conf, output_bound=obs_bound)

        dyna_env = DynamicsEnv(config=dynamics_env_conf, sess=sess, model=dynamics_model, init_env=real_env,
                               cost=cost_fn, done=done_fn)
        trpo = TrpoModel(config=trpo_config, action_bound=action_bound, obs_bound=obs_bound)

        sampler = FakeSampler(cost_fn=cost_fn, config=None, reference_trainer_env=player.env)
        ddpg_agent = TargetAgent(config=target_agent_config,
                                 real_env=real_env,
                                 cyber_env=dyna_env,
                                 model=trpo,
                                 sampler=sampler)

        env = TrainerEnv(config=trainer_env_conf,
                         cyber_env=dyna_env,
                         real_env=real_env,
                         target_agent=ddpg_agent,
                         test_env=test_env)

        baseline_trainer_model = FixedOutputModel(config=baseline_trainer_model_config)

        trainer_agent = BaselineTrainerAgent(config=baseline_trainer_agent_config,
                                             model=baseline_trainer_model,
                                             env=env)
        # END

        basic_list = [dynamics_model, dyna_env, trpo, ddpg_agent, env, trainer_agent, trainer_agent,
                      sampler]

        assemble_player = GamePlayer(config=player_config, env=env, agent=trainer_agent, basic_list=basic_list,
                                     ep_type=1, log_path=player.logger.log_dir + name)

        return assemble_player


def update_config_dict(config, dict):
    for key, value in dict.items():
        config.config_dict[key] = value


def update_state_action_space_by_env(real_env, dynamics_model_conf, dynamics_env_conf, ddpg_conf):
    config_list = [dynamics_model_conf, dynamics_env_conf, ddpg_conf]
    space_dict = {
        "STATE_SPACE": tuple(real_env.observation_space.shape),
        "ACTION_SPACE": tuple(real_env.action_space.shape),
    }
    for basic in config_list:
        update_config_dict(config=basic, dict=space_dict)
    # dynamics_model_conf.config_dict['OUTPUT_LOW'] = obs_bound[0]
    # dynamics_model_conf.config_dict['OUTPUT_HIGH'] = obs_bound[1]
    # ddpg_conf.config_dict['ACTION_LOW'] = action_bound[0]
    # ddpg_conf.config_dict['ACTION_HIGH'] = action_bound[1]
    pass


def update_name(config_list, name):
    for config in config_list:
        update_config_dict(config=config, dict={'NAME': config.config_dict['NAME'] + name})
