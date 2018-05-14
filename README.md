# Intelligent Trainer

## 1. How to use

### 1.1 Install everything
We use `Python3.5` and [Anaconda](https://www.anaconda.com/download/) to manage the packages, 
the required packages is listed at the file `package-list.txt`. 
So firstly, install `Anaconda 3.6 version`  if you don't have one. 

1.1.1 Build a new anaoconda env and install the packages:
```bash
conda create -
source activate intelligenttrainer
```
1.1.2 Install [mujoco 131](http://www.mujoco.org/) and [Mujoco-py](https://github.com/openai/mujoco-py).

For mujoco, you should install mjpro 131, and also register a license. 

For mujoco-py, run:
```bash
source activate intelligenttrainer
pip install mujoco-py==0.5.7
```

1.1.3 Install [OpenAI Gym](https://github.com/openai/gym/) and [OpenAI baselines](https://github.com/openai/baselines)

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
Firstly activate the anaconda environment:
```bash
source activate intelligenttrainer

```
1.2.1 To run the baseline experiments:
```bash
python testBaseline.py 
```

1.2.2 To run the intelligent trainer experiments:
```bash
python testIntelligent.py
```

1.2.3 Results and Visualize:
Every time you run experiments, the log file will be stored automatically in the `log/` directory.

`log/baselineTestLog` : the baseline experiments log 

`log/intelligentTestLog`: the intelligent trainer experiments log.

For each directory, it stored in each test cases sub directory 
and then named by the time you start to run the experiments.

Like `/log/baselineTestLog/MountainCarContinuous-v0/2018-05-14_17-15-13` 
means a baseline trainer experiments running on environment `MountainCarContinuous-v0` 
with time stamp `2018-05-14_17-15-13`

For each sub directory, it was structured in this way:
```bash
log/.../2018-05-14_17-15-13/config: the configuration file you used in this experiments.
log/.../2018-05-14_17-15-13/loss: record all training process information, like loss, reward etc.
log/.../2018-05-14_17-15-13/model: store all tensorflow model by the end of experiments.
log/.../2018-05-14_17-15-13/tf: a tensorboard file that store some training information which can be used to monitor the experiments
```

In `test/visualize.py`, we implement some utilities to visualize the results, this is also we generate the figure we used in our paper.
More documents about it will be done in the future.


## 2. Design of codes
### 2.1 A very simple UML figure about the desgin of our code.
![UML figure of our code](file:///C:/Users/luke%20dong/Dropbox/NIPS_intelligent_trainer_framework_code_document/UMLGraph/SimplifyModel.svg)
### 2.1 Some important classes