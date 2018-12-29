import gym
from src.core import Basic
from gym.core import Space
from src.env.util.step import Step


class BasicEnv(gym.Env, Basic):
    key_list = []

    def __init__(self, config):
        super(BasicEnv, self).__init__(config=config)
        self.action_space = Space()
        self.observation_space = Space()
        self.cost_fn = None
        self.step_count = 0
        self.stepper = Step(config=None)

    def step(self, action):
        return self.stepper.step(self, action)

    def reset(self):
        return None

    def init(self):
        print("%s init finished" % type(self).__name__)


if __name__ == '__main__':
    a = BasicEnv(config=1)
