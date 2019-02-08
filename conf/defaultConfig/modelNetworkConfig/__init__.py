from conf.modelNetworkConfig.ant import ANT_CONFIG
from conf.modelNetworkConfig.halfCheetah import HALFCHEETAH_CONFIG
from conf.modelNetworkConfig.humanoidStandup import HUMANOID_STANDUP_CONFIG
from conf.modelNetworkConfig.mountainCarContinuous import MOUNTAIN_CONFIG
from conf.modelNetworkConfig.pendulum import PENDULUM_CONFIG
from conf.modelNetworkConfig.swimmer import SWIMMER_CONFIG


MODEL_NET_WORK_CONFIG_DICT = {

    "Pendulum-v0": PENDULUM_CONFIG,
    "Ant-v1": ANT_CONFIG,
    "HumanoidStandup-v1": HUMANOID_STANDUP_CONFIG,
    "MountainCarContinuous-v0": MOUNTAIN_CONFIG,
    "Swimmer-v1": SWIMMER_CONFIG,
    "HalfCheetah": HALFCHEETAH_CONFIG
}

