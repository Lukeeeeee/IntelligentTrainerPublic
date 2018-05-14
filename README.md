# Intelligent Trainer

## 1. How to use

### 1.1 Install everything
We use `Python3.5` and `Anaconda` to manage the packages, 
the required packages is listed at the file `package-list.txt`.

1.1.1 Build a new anaoconda env and install the packages:
```bash
conda create -
source activate intelligenttrainer
```
1.1.2 Install [mujoco 131](http://www.mujoco.org/) and [Mujoco-py](https://github.com/openai/mujoco-py).
For mujoco, you should install mjpro 131, and also register a license. 
For mujoco-py
```bash
source activate intelligenttrainer
pip install mujoco-py==0.5.7
```

1.1.3 Install [OpenAI Gym](https://github.com/openai/gym/), [openAI baselines](https://github.com/openai/baselines)
We use specific version of gym which can support mjpro 131 and mujoco-py 0.5.7 due to compatiblity problem

Install the openai gym by:
```bash
source activate intelligenttrainer
git clone https://github.com/openai/gym
cd gym
git checkout 1d8565717206e54fca9e73ea1b3969948b464c3c
pip install -e .
```
Install openai baselines by following the official github.

1.1.4 


### 1.2 Run different experiments

## 2. Design of codes
### 2.1 Some important classes