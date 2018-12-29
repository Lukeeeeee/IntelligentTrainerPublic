import numpy as np
from conf.costFunction import COST_FUNCTION_PATH
from sklearn.preprocessing import PolynomialFeatures
import math
from conf.envBound import get_bound_file
from src.config.config import Config


class CostFunction(object):
    def __init__(self, parameters_file, obs_bound):
        self.par = Config.load_json(file_path=parameters_file)
        self.degree = int(self.par[0])
        self.coef = self.par[1: len(self.par) - 1]
        self.intec = self.par[len(self.par) - 1]
        self.poly_feature = PolynomialFeatures(degree=self.degree)
        self.obs_low = obs_bound['STATE_LOW']
        self.obs_high = obs_bound['STATE_HIGH']

    def __call__(self, state, action, next_state, *args, **kwargs):
        sample = []
        state_list = state.tolist()

        for i in range(len(list(state))):
            sample.append(state[i])
        for i in range(len(list(next_state))):
            sample.append(next_state[i])
        for i in range(len(list(action))):
            sample.append(action[i])
        sample = self.poly_feature.fit_transform(X=np.reshape(sample, newshape=[1, -1]))
        res = np.sum(np.multiply(sample, self.coef)) + self.intec
        return res


def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        #        print ("state.shape=", state.shape)
        #        print ("action.shape=", action.shape)
        #        print ("next_state.shape=", next_state.shape)

        heading_penalty_factor = 10
        scores = np.zeros((state.shape[0],))

        # dont move front shin back so far that you tilt forward
        front_leg = state[:, 5]
        my_range = 0.2
        scores[front_leg >= my_range] += heading_penalty_factor

        front_shin = state[:, 6]
        my_range = 0
        scores[front_shin >= my_range] += heading_penalty_factor

        front_foot = state[:, 7]
        my_range = 0
        scores[front_foot >= my_range] += heading_penalty_factor

        scores -= (next_state[:, 17] - state[:, 17]) / 0.01  # + 0.1 * (np.sum(action**2, axis=1))
        return -scores

    heading_penalty_factor = 10
    score = 0

    # dont move front shin back so far that you tilt forward

    front_leg = state[5]
    my_range = 0.2
    if front_leg >= my_range:
        score += heading_penalty_factor

    front_shin = state[6]
    my_range = 0
    if front_shin >= my_range:
        score += heading_penalty_factor

    front_foot = state[7]
    my_range = 0
    if front_foot >= my_range:
        score += heading_penalty_factor

    score -= (next_state[17] - state[17]) / 0.01  # + 0.1 * (np.sum(action**2))
    return -score


def continuous_mountain_car_reward_function(state, next_state, action):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    power = 0.0015

    position = state[0]
    velocity = state[1]
    force = min(max(action[0], -1.0), 1.0)

    velocity += force * power - 0.0025 * math.cos(3 * position)
    if velocity > max_speed:
        velocity = max_speed
    if velocity < -max_speed:
        velocity = -max_speed
    position += velocity
    if position > max_position:
        position = max_position
    if position < min_position:
        position = min_position

    if position == min_position and velocity < 0:
        velocity = 0

    done = bool(position >= goal_position)
    reward = 0
    if done:
        reward = 100.0
    reward -= math.pow(action[0], 2) * 0.1

    return reward


def reacher_reward_function(state, next_state, action):
    vec = state[-3:]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum()
    reward = reward_dist + reward_ctrl
    return reward


def CFD_reward_function(state, next_state, action):
    temp_reading = next_state[-5:]

    reward = 2.0 - (action[0] / 100.0 ** 3 + action[1] / 100.0 ** 3) - 10.0 * min(np.max(temp_reading) - 35.0, 0)
    return reward


def inverted_pendulum_reward_function(state, next_state, action):
    return 1.0


def swimmer_reward_function(state, next_state, action):
    s_vel = state[-5:]
    pass

# def pendulum_reward_function(state, next_state, action):
#     max_torque = 2.
#
#     th = math.acos(state[0])
#     th2 = math.asin(state[1])
#     assert math.fabs(th - th2) <= 0.00001
#     print(th, th2)
#     thdot = state[2]
#
#     u = np.clip(action, -max_torque, max_torque)[0]
#
#     costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
#
#     return costs


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


COST_FUNCTION_ENV_DICT = {
    "Pendulum-v0": CostFunction(COST_FUNCTION_PATH + '/Pendulum-v0_cost_function_parameters.json',
                                obs_bound=Config.load_json(
                                    file_path=get_bound_file(env_name='Pendulum-v0'))),
    "Ant-v1": CostFunction(COST_FUNCTION_PATH + '/Ant-v1_cost_function_parameters.json',
                           obs_bound=Config.load_json(
                               file_path=get_bound_file(env_name='Ant-v1'))),
    "HumanoidStandup-v1": CostFunction(COST_FUNCTION_PATH + '/HumanoidStandup-v1_cost_function_parameters.json',
                                       obs_bound=Config.load_json(
                                           file_path=get_bound_file(env_name='HumanoidStandup-v1'))),
    "MountainCarContinuous-v0": continuous_mountain_car_reward_function,
    "Swimmer-v1": CostFunction(COST_FUNCTION_PATH + '/Swimmer-v1_cost_function_parameters.json',
                               obs_bound=Config.load_json(file_path=get_bound_file(env_name='Swimmer-v1'))),
    "HalfCheetah": cheetah_cost_fn,
    "Reacher-v1": reacher_reward_function,
    "CFD": CFD_reward_function,
    "InvertedPendulum-v1": inverted_pendulum_reward_function
}
