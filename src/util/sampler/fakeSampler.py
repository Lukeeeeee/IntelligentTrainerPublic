from src.core import Basic
import numpy as np
from src.config.config import Config
from config.key import CONFIG_KEY
import tensorflow as tf
from src.util.sampler.sampler import Sampler
from src.util.sampler.sampler import SamplerData
from src.util.sampler.intelligentSampler import IntelligentSampler
from copy import deepcopy


class FakeSampler(Sampler):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/fakeSamplerKey.json')

    def __init__(self, config, cost_fn, reference_trainer_env):
        super().__init__(config=config, cost_fn=cost_fn)
        self.reference_trainer_env = reference_trainer_env

    def sample(self, env, agent, sample_count, store_flag=False, agent_print_log_flag=False, reset_Flag=True):
        if agent.status == agent.status_key['TEST'] or agent.env_status == agent.config.config_dict[
            'CYBER_ENVIRONMENT_STATUS']:
            res = super().sample(agent=agent,
                                 env=env,
                                 sample_count=sample_count,
                                 store_flag=store_flag,
                                 agent_print_log_flag=agent_print_log_flag,
                                 reset_Flag=reset_Flag)
            return res

        if agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
            sample_data, enough_flag = self.reference_trainer_env.target_agent.model.return_most_recent_sample(
                sample_count=sample_count,
                env_status=self.reference_trainer_env.target_agent.config.config_dict['REAL_ENVIRONMENT_STATUS'])
            if enough_flag is False:
                raise ValueError('Real env data is not enough')

            sample_record = SamplerData()
            for i in range(sample_count):
                state, new_state, action, re, done = self._return_index_sample(sample_data=sample_data, index=i)
                if not isinstance(done, bool):
                    if done[0] == 1.0:
                        done = True
                    else:
                        done = False
                if self.cost_fn:
                    reward = self.cost_fn(state=state, action=action, next_state=new_state)
                else:
                    reward = re
                from src.agent.targetAgent.targetAgent import TargetAgent
                if isinstance(agent, TargetAgent):
                    if agent.status == agent.status_key['TEST'] and \
                            agent.env_status == agent.config.config_dict['REAL_ENVIRONMENT_STATUS']:
                        pass
                    else:
                        re = reward

                self.step_count_per_episode += 1
                agent.env_sample_count += 1
                if store_flag is True:
                    agent.store_one_sample(state=state,
                                           action=action,
                                           next_state=new_state,
                                           reward=re,
                                           done=done)

                self.data.append(state=state,
                                 action=action,
                                 new_state=new_state,
                                 done=done,
                                 reward=re)

                sample_record.append(state=state,
                                     action=action,
                                     reward=re,
                                     new_state=new_state,
                                     done=done)

                self.log_every_step(agent=agent,
                                    reward=re)
                if done is True:
                    tmp_state_set = self.state_set + [self.new_state_set[-1]]
                    self.log_every_episode(agent=agent,
                                           average_reward=self.cumulative_reward / self.step_count_per_episode,
                                           reward=self.cumulative_reward,
                                           state_set=tmp_state_set,
                                           action_set=self.action_set,
                                           agent_print_log_flag=agent_print_log_flag)
                    agent.log_queue.queue.clear()

            return sample_record

        else:
            raise ValueError('Environment status wrong')

    def train(self, *args, **kwargs):
        pass

    def print_log_queue(self, status):
        pass

    def _return_index_sample(self, sample_data, index):
        return (sample_data.state_set[index],
                sample_data.new_state_set[index],
                sample_data.action_set[index],
                sample_data.reward_set[index],
                sample_data.done_set[index])
