import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from src.env.util.cost import cheetah_cost_fn


class HalfCheetahEnvNew(mujoco_env.MujocoEnv, utils.EzPickle):

    name = 'HalfCheetah'

    def __init__(self, cost=cheetah_cost_fn):
        self.cost = cost
        self.step_count = 0
        self.name = 'HalfCheetah'
        self._elapsed_steps = 0
        self._max_episode_steps = 1000

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self._elapsed_steps += 1
        # xposbefore = self.model.data.qpos[0, 0]
        prev_obs = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore) / self.dt
        # reward = reward_ctrl + reward_run
        # change this part
        reward = self.cost(state=prev_obs, action=action, next_state=ob)
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        else:
            done = False
        if done is True:
            self.reset()

        return ob, reward, done, None

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            # self.get_body_comvel("torso").flat,
        ])

    def reset_model(self):
        self._elapsed_steps = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def print_log_queue(self):
        pass

    def reset(self):
        # print("%s reset finished" % type(self).__name__)
        return self.reset_model()

    def init(self):
        print("%s init finished" % type(self).__name__)


if __name__ == '__main__':
    a = HalfCheetahEnvNew()
    res, reward, done, info = a.step(action=a.action_space.sample())
    pass
