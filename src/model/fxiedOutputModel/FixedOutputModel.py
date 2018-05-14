from src.model.model import Model
from src.config.config import Config
from config.key import CONFIG_KEY
import numpy as np


class FixedOutputModel(Model):
    key_list = Config.load_json(file_path=CONFIG_KEY + '/fixedOutputModelKey.json')

    def __init__(self, config):
        super(FixedOutputModel, self).__init__(config)

    def predict(self, sess=None, state=None):
        # baseline mode use fixed policy currently
        action = [0 for i in range(self.config.config_dict['ACTION_SPACE'][0])]
        action[0] = self.config.config_dict['F1']
        action[1] = self.config.config_dict['PROB_SAMPLE_ON_REAL']
        action[2] = self.config.config_dict['PROB_TRAIN_ON_REAL']
        # action[3] = 0.0
        # action[3] = self.config.config_dict['S']

        return np.array(action)

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def print_log_queue(self, status):
        pass


if __name__ == '__main__':
    from config import CONFIG

    conf = Config(standard_key_list=FixedOutputModel.key_list)
    conf.load_config(path=CONFIG + '/baselineTrainerModelTestConfig.json')
    a = FixedOutputModel(config=conf)
    print(a.predict(state=1))
