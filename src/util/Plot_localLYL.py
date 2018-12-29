import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import math as M

from src.util.plotter import Plotter


##half Cheetah
path_list = ["/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_19-19-57_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_19-20-20_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_19-20-47_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_19-26-07_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_19-27-15_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_21-06-01_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_22-06-49_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-13_22-58-35_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-14_00-29-19_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-14_01-27-59_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-14_01-44-25_INTEL_v2_2000",]

###pen
# path_list = ["/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_02-41-19_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_03-57-33_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_04-36-52_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_05-49-56_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_06-54-52_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_07-37-43_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_08-02-12_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_08-52-06_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_09-35-56_INTEL_v2_2000",
#              "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-14_09-56-20_INTEL_v2_2000",]

### car
path_list = ["/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_10-39-30_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_10-39-52_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_11-15-54_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_11-20-13_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_11-57-54_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_12-02-07_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_12-37-42_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_12-42-54_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_13-20-19_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-14_13-23-55_INTEL_v2_2000",]

##reacher
path_list = ["/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_02-38-49_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_05-36-05_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_07-50-32_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_07-51-09_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_07-58-20_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_14-15-44_INTEL_v2_2000",
             "/home/liyuanl/MRL_New/NewCode/intelligenttrainerframework/log/intelligentTestLog/Reacher-v1/2018-05-14_11-17-47_INTEL_v2_2000",
             ]
Plotter.plot_mean_multiply_target_agent_reward(path_list, path_list)
plt.show()