from collections import deque
import numpy as np


class DynamicsEnvironmentMemory(object):
    def __init__(self):
        self.data = deque(maxlen=2000)

    def sample(self, batch_size):
        batch_size = min(len(self.data), batch_size)
        batch_idxs = np.random.random_integers(len(self.data) - 2, size=batch_size)

        batch_set = []
        for idx in batch_idxs:
            batch_set.append(self.data[idx])
        obs0 = []
        obs1 = []
        reward = []
        action = []
        done = []
        delta = []
        for data in batch_set:
            obs0.append(data['obs0'])
            obs1.append(data['obs1'])
            reward.append(data['reward'])
            action.append(data['action'])
            done.append(data['terminal1'])
            delta.append(data['delta_state'])
        return {
            'obs0': obs0,
            'obs1': obs1,
            'action': action,
            'reward': reward,
            'terminal1': done,
            'delta': delta
        }

    def append(self, sample):
        self.data.append(sample)
