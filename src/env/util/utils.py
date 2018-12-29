import gym
import socket
import time
import config as cfg

ENV_MAX_EPOSIDE_STEPS = {
    "MountainCarContinuous-v0": 999,
    "Pendulum-v0": 200,
    "Ant-v1": 1000,
    "HumanoidStandup-v1": 1000,
    "Swimmer-v1": 1000,
    "HalfCheetah": 1000,
    "Reacher-v1": 50
}


def sendMsg(msg, serverIp='155.69.146.42', tcpPort=34526, waittime=100):
    redo = 0
    redoc = 0
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(waittime)
        s.connect((serverIp, tcpPort))
    except socket.error as err:
        print("socket creation failed with error %s" % (err))
        redo = 1
        redoc += 1
        print("retying send: ", redoc)
        return redo
    time.sleep(1)
    s.send(msg.encode())
    print("Sent successful")
    s.close()
    return redo


def recvMsg(port=23445, buf=1024, waittime=100):
    host = '0.0.0.0'
    addr = (host, port)

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(addr)
    serversocket.listen(20)
    serversocket.settimeout(waittime)
    ret = -1
    is_timeout = 0
    data = ''
    try:
        print("CFDEnv is listening for returns\n")
        clientsocket, clientaddr = serversocket.accept()
        print("Accepted connection from: ", clientaddr)
        while True:
            data = clientsocket.recv(buf)
            if not data:
                break
            print("data=", "_" + str(data) + "_")
            if len(str(data)) > 0:
                break
        clientsocket.close()
    except socket.timeout:
        is_timeout = 1
        print("Time out detected")

    if data == '':
        redo = 1
        return [ret, redo]

    # data = data.decode('utf-8')
    data = str(data)
    print("raw data=", str(data))
    redo = 0
    if len(data) > 0:
        if data[0] == 'b':
            data = data[1:]
        if data[-1] == 'b':
            data = data[:-1]
        if data[0] == "'":
            data = data[1:]
        if data[-1] == "'":
            data = data[:-1]
    print("decoded data=", data)
    if len(data) > 0:
        print("decoded data=", data)
        ret = [float(s) for s in data.split(' ')]
    elif is_timeout == 1:
        redo = 1
    return [ret, redo]


class EnvMakerWrapper(object):
    def __init__(self, name):
        self.id = name
        self.name = name
        self.__class__.__name__ = self.id

    def __call__(self, *args, **kwargs):
        return gym.make(self.id)

        # if 'SWIMMER_HORIZON_50' in cfg.config_dict and cfg.config_dict['SWIMMER_HORIZON_50'] is True and self.id == 'Swimmer-v1':
        #     a = gym.make(self.id)
        #     a._max_episode_steps = 50
        #     return a
        # else:
        #     return gym.make(self.id)


from src.env.halfCheetahEnv.halfCheetahEnv import HalfCheetahEnvNew
from src.env.CFDEnv.CFDEnv import CFDEnv

GAME_ENV_NAME_DICT = {
    "Pendulum-v0": EnvMakerWrapper('Pendulum-v0'),
    "Ant-v1": EnvMakerWrapper('Ant-v1'),
    "HumanoidStandup-v1": EnvMakerWrapper('HumanoidStandup-v1'),
    "MountainCarContinuous-v0": EnvMakerWrapper('MountainCarContinuous-v0'),
    "Swimmer-v1": EnvMakerWrapper('Swimmer-v1'),
    "HalfCheetah": HalfCheetahEnvNew,
    "Reacher-v1": EnvMakerWrapper('Reacher-v1'),
    "CFD": CFDEnv,
    "InvertedPendulum-v1": EnvMakerWrapper('InvertedPendulum-v1')
    # "Hopper-v1": EnvMakerWrapper('Hopper-v1')
}

pendulum_instance = GAME_ENV_NAME_DICT['Pendulum-v0']()
# ant_instance = GAME_ENV_NAME_DICT['Ant-v1']()
# humanoid_standup_instance = GAME_ENV_NAME_DICT['HumanoidStandup-v1']()
mountain_car_continuous_instance = GAME_ENV_NAME_DICT['MountainCarContinuous-v0']()
swimmer_instance = GAME_ENV_NAME_DICT['Swimmer-v1']()
half_cheetah_instance = GAME_ENV_NAME_DICT['HalfCheetah']()
reacher_instance = GAME_ENV_NAME_DICT['Reacher-v1']()
# CFD_instance = GAME_ENV_NAME_DICT['CFD']()
inverted_pendulum_instance = GAME_ENV_NAME_DICT['InvertedPendulum-v1']()


# hopper_instance = GAME_ENV_NAME_DICT['Hopper-v1']()


def pendulum_get_state(env):
    if isinstance(env, type(pendulum_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


# def ant_get_state(env):
#     if isinstance(env, type(ant_instance)) is True:
#         return env.unwrapped._get_obs()
#     else:
#         raise ValueError('Wrong type of environment to get state')

#
# def humanoid_standup_get_state(env):
#     if isinstance(env, type(humanoid_standup_instance)) is True:
#         return env.unwrapped._get_obs()
#     else:
#         raise ValueError('Wrong type of environment to get state')


def mountain_car_continuous_get_state(env):
    if isinstance(env, type(mountain_car_continuous_instance)) is True:
        return env.unwrapped.state
    else:
        raise ValueError('Wrong type of environment to get state')


def swimmer_get_state(env):
    if isinstance(env, type(swimmer_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def half_cheetah_get_state(env):
    if isinstance(env, type(half_cheetah_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


def reacher_get_state(env):
    if isinstance(env, type(reacher_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


# def CFD_get_state(env):
#     if isinstance(env, type(CFD_instance)) is True:
#         return env.unwrapped._get_obs()
#     else:
#         raise ValueError('Wrong type of environment to get state')


def inverted_pendulum_get_state(env):
    if isinstance(env, type(inverted_pendulum_instance)) is True:
        return env.unwrapped._get_obs()
    else:
        raise ValueError('Wrong type of environment to get state')


# def hopper_get_state(env):
#     if isinstance(env, type(hopper_instance)) is True:
#         return env.unwrapped._get_obs()
#     else:
#         raise ValueError('Wrong type of environment to get state')


GET_STATE_FUNCTION_DICT = {
    "Pendulum-v0": pendulum_get_state,
    # "Ant-v1": ant_get_state,
    # "HumanoidStandup-v1": humanoid_standup_get_state,
    "MountainCarContinuous-v0": mountain_car_continuous_get_state,
    "Swimmer-v1": swimmer_get_state,
    "HalfCheetah": half_cheetah_get_state,
    "Reacher-v1": reacher_get_state,
    # "CFD": CFD_get_state,
    "InvertedPendulum-v1": inverted_pendulum_get_state,
    # "Hopper-v1": hopper_get_state
}


def make_env(id):
    env = GAME_ENV_NAME_DICT[id]()
    env.get_state = GET_STATE_FUNCTION_DICT[id]
    return env
