import numpy as np
import math
import json


def compute_best_eps_reward(test_file, eps_size=100):
    average_reward = 0.0
    reward_list = []
    real_sample_used_list = []
    cyber_sample_used_list = []
    count = 0
    with open(file=test_file, mode='r') as f:
        train_data = json.load(fp=f)
        for sample in train_data:
            # if count % 2 == 0:
            #     print(sample['REWARD_SUM'])
            #     pass
            # else:
            #     reward_list.append(sample['REWARD_SUM'])
            #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

            reward_list.append(sample['REWARD_SUM'])
            real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])
            #
            # if sample['REWARD_SUM'] >= 100:
            #     print(sample['REWARD_SUM'])
            #     pass
            # else:
            #     reward_list.append(sample['REWARD_SUM'])
            #     real_sample_used_list.append(sample['REAL_SAMPLE_COUNT'])
            #     cyber_sample_used_list.append(sample['CYBER_SAMPLE_COUNT'])

            count += 1

    min_reward = -100000.0
    pos = 0
    average_reward_list = []
    for i in range(len(reward_list) - eps_size + 1):
        average_reward = sum(reward_list[i: i + eps_size]) / eps_size
        # if average_reward >= 73.0:
        #     print(i, real_sample_used_list[i], cyber_sample_used_list[i], average_reward)
        average_reward_list.append(average_reward)
        # if real_sample_used_list[i] > max_real_sample * 0.5:
        #     if average_reward > min_reward:
        #         min_reward = average_reward
        #         pos = i
        if average_reward > min_reward:
            min_reward = average_reward
            pos = i

    # reward_list.sort(reverse=True)
    # average_reward = sum(reward_list[0:100]) / 100.0
    # print(average_reward)

    return reward_list, real_sample_used_list, cyber_sample_used_list, pos, min_reward, average_reward_list


def print_best_eps_reward(file_path, eps_size):
    reward_list, real_sample_used_list, cyber_sample_used_list, pos, min_reward, average_reward_list = \
        compute_best_eps_reward(test_file=file_path, eps_size=eps_size)

    max_real_sample = max(real_sample_used_list)
    for i in range(len(average_reward_list)):
        print(i, average_reward_list[i])

    print("max reward ", max(reward_list))

    print("100 episode reward %f, at real sample count %d, cyber sample count %d\nmax real sample is %d\n" %
          (min_reward, real_sample_used_list[pos], cyber_sample_used_list[pos], max_real_sample))


if __name__ == '__main__':
    print_best_eps_reward(
        "/home/dls/CAP/intelligenttrainerframework/log/intelligentTestLog/Pendulum-v0/2018-05-02_20-16-25_new/loss/TargetAgent_test_.log",
        eps_size=100)
