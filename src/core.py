import tensorflow as tf
import time
from log.baselineTestLog import LOG
import os
from src.config.config import Config
from conf.key import CONFIG_KEY
import numpy as np
import random
import queue
import json
import easy_tf_log
from log.intelligentTestLog import INTEL_LOG


class Basic(object):
    key_list = []
    status_key = {'TRAIN': 0, 'TEST': 1}

    def __init__(self, config):
        self.config = config
        self.name = type(self).__name__

        self._train_log_file = self.name + '_train_.log'
        self._test_log_file = self.name + '_test_.log'

        self._train_log_queue = queue.Queue(maxsize=int(1e10))
        self._test_log_queue = queue.Queue(maxsize=int(1e10))

        self._train_log_print_count = 0
        self._test_log_print_count = 0

        self._train_log_file_content = []
        self._test_log_file_content = []

        self._status = Basic.status_key['TRAIN']

        self._log_file = None
        self._log_queue = None
        self._log_print_count = None
        self._log_file_content = None

    def print_log_queue(self, status):
        self.status = status
        while self.log_queue.qsize() > 0:
            content = self.log_queue.get()
            self.log_file_content.append({'INDEX': self.log_print_count, 'LOG': content})
            print(content)
            self.log_print_count += 1

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_value):
        if new_value != Basic.status_key['TRAIN'] and new_value != Basic.status_key['TEST']:
            raise KeyError('New Status: %d did not existed' % new_value)

        if self._status == new_value:
            return
        self._status = new_value

    @property
    def log_file(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_queue(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_queue
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_queue
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_file_content(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_file_content
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_file_content
        raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def log_print_count(self):
        if self._status == Basic.status_key['TRAIN']:
            return self._train_log_print_count
        elif self._status == Basic.status_key['TEST']:
            return self._test_log_print_count
        raise KeyError('Current Status: %d did not existed' % self._status)

    @log_print_count.setter
    def log_print_count(self, new_val):
        if self._status == Basic.status_key['TRAIN']:
            self._train_log_print_count = new_val
        elif self._status == Basic.status_key['TEST']:
            self._test_log_print_count = new_val
        else:
            raise KeyError('Current Status: %d did not existed' % self._status)

    @property
    def current_status(self):
        if self._status == Basic.status_key['TRAIN']:
            return 'TRAIN'
        elif self._status == Basic.status_key['TEST']:
            return 'TEST'
        else:
            raise KeyError('Current Status: %d did not existed' % self._status)


class Logger(object):

    def __init__(self, prefix=None, log=LOG, log_path=None, log_path_end=None):
        if log_path is not None:
            self._log_dir = log_path
        else:
            self._log_dir = log + '/' + prefix + '/' + time.strftime("%Y-%m-%d_%H-%M-%S")
        if log_path_end:
            self._log_dir = self._log_dir + log_path_end
        self._config_file_log_dir = None
        self._loss_file_log_dir = None
        self._model_file_log_dir = None
        if os.path.exists(self._log_dir):
            raise FileExistsError('%s path is existed' % self._log_dir)

    @property
    def log_dir(self):
        if os.path.exists(self._log_dir) is False:
            os.makedirs(self._log_dir)
        return self._log_dir

    @property
    def config_file_log_dir(self):
        self._config_file_log_dir = os.path.join(self.log_dir, 'conf')
        if os.path.exists(self._config_file_log_dir) is False:
            os.makedirs(self._config_file_log_dir)
        return self._config_file_log_dir

    @property
    def loss_file_log_dir(self):
        self._loss_file_log_dir = os.path.join(self.log_dir, 'loss')
        if os.path.exists(self._loss_file_log_dir) is False:
            os.makedirs(self._loss_file_log_dir)
        return self._loss_file_log_dir

    @property
    def model_file_log_dir(self):
        self._model_file_log_dir = os.path.join(self.log_dir, 'model/')
        if os.path.exists(self._model_file_log_dir) is False:
            os.makedirs(self._model_file_log_dir)
        return self._model_file_log_dir

    def out_to_file(self, file_path, content):
        with open(file_path, 'w') as f:
            # TODO how to modify this part
            for dict_i in content:
                for key, value in dict_i.items():
                    if isinstance(value, np.generic):
                        dict_i[key] = value.item()
            json.dump(content, fp=f, indent=4, sort_keys=True)


if __name__ == '__main__':
    pass
