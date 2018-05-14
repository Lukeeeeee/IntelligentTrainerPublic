from src.config.config import Config

key_list = [
    "intelligentRandomTrainerAgentKey.json",
    "dynamicsEnvKey.json",
    "baselineTrainerAgentKey.json",
    "trainerEnvKey.json",
    "dynamicsEnvMlpModelKey.json",
    "fixedOutputModelKey.json",
    "fakeSamplerKey.json",
    "intelligentTrainerAgentKey.json",
    "ddpgModelKey.json",
    "trpoModelKey.json",
    "intelligentSamplerKey.json",
    "dqnModelKey.json",
    "baselineTrainerEnvKey.json",
    "reinforceModelKey.json",
    "gamePlayerKey.json",
    "ddpgAgentKey.json"
]

config_file_name = [
    "intelligentTrainerAgentTestConfig.json",
    "dynamicsEnvTestConfig.json",
    "gamePlayerTestConfig.json",
    "intelligentTrainerModelTestConfig.json",
    "baselineTrainerAgentTestConfig.json",
    "intelligentSamplerConfig.json",
    "ddpgAgentTestConfig.json",
    "baselineTrainerEnvTestConfig.json",
    "trainerEnvTestConfig.json",
    "baselineTrainerModelTestConfig.json"
]

model_file_name = [
    "dynamicsEnvMlpModelTestConfig.json",
    "targetModelTestConfig.json"
]

trpo_key = {
    "dynamicsEnvKey.json": "dynamicsEnvTestConfig.json",
    "baselineTrainerAgentKey.json": "baselineTrainerAgentTestConfig.json",
    "trainerEnvKey.json": "trainerEnvTestConfig.json",
    "dynamicsEnvMlpModelKey.json": "dynamicsEnvMlpModelTestConfig.json",
    "fixedOutputModelKey.json": "baselineTrainerModelTestConfig.json",
    "intelligentTrainerAgentKey.json": "intelligentTrainerAgentTestConfig.json",
    "trpoModelKey.json": "targetModelTestConfig.json",
    "intelligentSamplerKey.json": "intelligentSamplerConfig.json",
    "dqnModelKey.json": "intelligentTrainerModelTestConfig.json",
    "baselineTrainerEnvKey.json": "baselineTrainerEnvTestConfig.json",
    "gamePlayerKey.json": "gamePlayerTestConfig.json",
    "ddpgAgentKey.json": "ddpgAgentTestConfig.json"
}

ddpg_key = {
    "dynamicsEnvKey.json": "dynamicsEnvTestConfig.json",
    "baselineTrainerAgentKey.json": "baselineTrainerAgentTestConfig.json",
    "trainerEnvKey.json": "trainerEnvTestConfig.json",
    "dynamicsEnvMlpModelKey.json": "dynamicsEnvMlpModelTestConfig.json",
    "fixedOutputModelKey.json": "baselineTrainerModelTestConfig.json",
    "intelligentTrainerAgentKey.json": "intelligentTrainerAgentTestConfig.json",
    "ddpgModelKey.json": "targetModelTestConfig.json",
    "intelligentSamplerKey.json": "intelligentSamplerConfig.json",
    "dqnModelKey.json": "intelligentTrainerModelTestConfig.json",
    "baselineTrainerEnvKey.json": "baselineTrainerEnvTestConfig.json",
    "gamePlayerKey.json": "gamePlayerTestConfig.json",
    "ddpgAgentKey.json": "ddpgAgentTestConfig.json"
}
from config.key import CONFIG_KEY
import os
import json


def clear(config_path, model_path, key_dict, save_flag=False):
    for key, val in key_dict.items():
        conf = Config(standard_key_list=Config.load_json(file_path=os.path.join(CONFIG_KEY, key)))
        if val in model_file_name:
            path = model_path
        else:
            path = config_path
        path = os.path.join(path, val)
        print(key, path)
        conf.load_config(path=path)
        new_config_dict = {}
        for key, val in conf.config_dict.items():
            if key not in conf.standard_key_list:
                print(key, val)
            else:
                new_config_dict[key] = val
        if save_flag is True:
            with open(file=path, mode='w') as f:
                json.dump(new_config_dict, fp=f, indent=4, sort_keys=True)


if __name__ == '__main__':
    clear(config_path='/home/dls/CAP/intelligenttrainerpublic/config/configSet_Swimmer',
          model_path='/home/dls/CAP/intelligenttrainerpublic/config/configSet_Swimmer/modelNetworkConfig/swimmer',
          key_dict=trpo_key,
          save_flag=False)
