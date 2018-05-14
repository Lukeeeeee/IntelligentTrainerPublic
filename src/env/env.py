import gym
from src.core import Basic
from gym.core import Space


class BasicEnv(gym.Env, Basic):
    key_list = []

    def __init__(self, config):
        super(BasicEnv, self).__init__(config=config)
        self.action_space = Space()
        self.observation_space = Space()
        self.cost_fn = None
        self.step_count = 0

    def step(self, action):
        self.step_count += 1

    def reset(self):
        # print("%s reset finished" % type(self).__name__)
        return None

    def init(self):
        print("%s init finished" % type(self).__name__)


if __name__ == '__main__':
    a = BasicEnv(config=1)
