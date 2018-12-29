from src.env.env import BasicEnv
from gym.spaces.box import Box
import numpy as np
import copy
import time
from src.env.util.cost import CFD_reward_function
from src.env.util.utils import recvMsg, sendMsg
import pandas
from src.env import SRC_ENV_PATH
import os

'''
README
CFDEnv

This is the python script part.
The whole environment includes three parts:
1. Python API (this)
2. Matlab script ()
3. 6Sigma Simulation software.

Working flow:
1. In the CFD server, Start Matlab script
2. In the CFD server, switch to Simulation software, do move the mouse, as it relies on automatic mouse click to start and end the simulation
3. Start the Python API.

CFD server address:
IP: 155.69.146.42
Username: administrator
Password: ntuCAP123

Log in this server to check running state.
If the CFD software stuck, restart the matlab main script. If the CFD software is in Solving page, click stop/close at the right bottom corner.

 
Matlab script path:
C:\Program Files\6SigmaSDKRelease12\matlab\Examples\AliToyTest_lyl\SimpleTestCase\SteadyState\SolverExchange\

Main script:
SocketBasedMain.m
Make a copy of this file, then change the IP adress in line 46 to your IP adress.

To start the script:
press green trangle run button;
To stop script:
Ctrl + C

'''


class CFDEnv(BasicEnv):
    def __init__(self, cost=CFD_reward_function):
        super(CFDEnv, self).__init__(config=None)
        trace_name = []
        for i in range(1, 40):
            trace_name.append('Rack Power %02d' % i)
        data = pandas.read_csv(os.path.join(SRC_ENV_PATH, "CFDEnv/load_trace.csv"),
                               names=trace_name)
        load_trace = []
        for col in trace_name:
            load_trace.append(np.array(data[col][1:].astype(float)))
        self.load_trace = np.array(load_trace)
        min_load = self.load_trace.min()
        max_load = self.load_trace.max()

        self.load_trace = (self.load_trace - min_load) / (max_load - min_load) * 100.

        self.trace_sample = []

        self._max_step = 10
        self.cost = cost

        for i in range(self.load_trace.shape[0]):
            for j in range(self.load_trace.shape[1] - self._max_step - 1):
                self.trace_sample.append(self.load_trace[i][j: j + self._max_step + 1])

        # self.rack_num = self.load_trace.shape[0]

        self.action_space = Box(low=np.asarray([0.0, 0.0]),
                                high=np.asarray([100.0, 100.0]),
                                shape=None)

        # 5 dimension load + 5 dimension temperature
        self.observation_space = Box(low=np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 20.0, 20.0, 20.0, 20.0, 20.0]),
                                     high=np.asarray([100.0, 100.0, 100.0, 100.0, 100.0, 60.0, 60.0, 60.0, 60.0, 60.0]),
                                     shape=None)
        self._load_index = np.random.randint(low=0, high=len(self.trace_sample), size=5)
        self._obs = self.observation_space.sample()
        for i in range(5):
            self._obs[i] = self.trace_sample[self._load_index[i]][0]

        # TODO Modify the generator to different random process to get the load

    def step(self, action):
        super().step(action=action)
        # Encode load and action info into a csv format string

        # For random generate load obs, use a random number from a uniform distribution load as
        # the parameters of a random process and generate the new load step by step

        simulation_msg = "TimeStamp,E3_bottom,E3_middle,E3_top,E5_bottom,E5_top,ACU01,ACU02\n"
        simulation_msg += str(int(time.time())) + ","
        for i in range(5):
            simulation_msg += str(self._obs[i]) + ','
        simulation_msg += str(action[0]) + ','
        simulation_msg += str(action[1]) + '\n'
        redo = 1
        redo_count = 0
        temp_reading = None

        while redo == 1:
            if redo_count > 0:
                print("Retrying simulation: ", redo_count)
            # TODO for debug only
            redo = sendMsg(simulation_msg)
            # redo = 0
            if redo == 1:
                time.sleep(10)
                redo_count += 1
                continue
            # TODO for debug only
            [temp_reading, redo] = np.asarray(recvMsg(waittime=200))
            # temp_reading = self.observation_space.sample()[-5:]
            # redo = 0
            redo_count += 1
            if redo == 1:
                time.sleep(10)
        pre_obs = copy.deepcopy(self._get_obs())
        # update obs
        for i in range(5):
            self._obs[i] = self.trace_sample[self._load_index[i]][self.step_count+1]
        self._obs[-5:] = temp_reading

        obs = self._get_obs()
        reward = self.cost(state=pre_obs, next_state=obs, action=action)

        self.step_count = self.step_count + 1
        done = self.step_count >= self._max_step
        if done is True:
            self.reset()
        info = []
        # print("CFD One step done.")
        return obs, reward, done, info

    def _get_obs(self):
        return self._obs

    def reset(self):
        super().reset()
        self.step_count = 0
        self._load_index = np.random.randint(low=0, high=len(self.trace_sample), size=5)
        self._obs = self.observation_space.sample()
        for i in range(5):
            self._obs[i] = self.trace_sample[self._load_index[i]][0]
        return self._obs

    def init(self):
        super().init()

    def close(self):
        print('Close the CFD environment')

    def configure(self):
        pass

    def seed(self, seed=None):
        pass

    def get_state(self):
        return self._get_obs()


if __name__ == '__main__':
    a = CFDEnv()
    reward_t = []
    for i in range(20000):
        obs, reward, done, info = a.step(action=a.action_space.sample())
        print("obs=", obs)
        print("reward=", reward)
        reward_t.append(reward)
    print("total reward=", reward_t)
    print("mean, std, max, min = ", np.mean(reward_t), np.std(reward_t), np.max(reward_t), np.min(reward_t))
