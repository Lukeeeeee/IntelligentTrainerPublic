import numpy as np
import math


def continuous_mountain_car_done_function(state, next_state, action):
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

    return done


def inverted_pendulum_done_function(state, next_state, action):
    notdone = np.isfinite(next_state).all() and (np.abs(next_state[1]) <= .2)
    return not notdone


DONE_FUNCTION_ENV_DICT = {
    "MountainCarContinuous-v0": continuous_mountain_car_done_function,
    "Pendulum-v0": None,
    "Ant-v1": None,
    "HumanoidStandup-v1": None,
    "Swimmer-v1": None,
    "HalfCheetah": None,
    "Reacher-v1": None,
    "CFD": None,
    "InvertedPendulum-v1": inverted_pendulum_done_function
}
