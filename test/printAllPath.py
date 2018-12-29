import os
import json
from src.core import Config


def _check_log_integrity(log_path_dir):
    max_step = Config.load_json(os.path.join(log_path_dir, 'conf', 'GamePlayer.json'))['MAX_REAL_ENV_SAMPLE']
    if not os.path.exists(os.path.join(log_path_dir, 'loss')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    if not os.path.exists(os.path.join(log_path_dir, 'loss', 'TargetAgent_train_.log')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False

    agent_train_log = Config.load_json(os.path.join(log_path_dir, 'loss', 'TargetAgent_train_.log'))
    if len(agent_train_log) == 0:
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    for sample in agent_train_log:
        if sample['REAL_SAMPLE_COUNT'] >= max_step:
            return True
    print("Log %s, did not finish, omitted" % (log_path_dir))
    return False


def print_all_dir_name(father_dir, env_name, contain_char=None, save_file=None, not_contain_char=None):
    dir_list = []
    for dir in os.listdir(father_dir):
        flag = None
        if contain_char:
            if contain_char in dir:
                flag = True
        if not_contain_char:
            for ch in not_contain_char:
                if ch in dir:
                    flag = False
        if flag is True:
            if _check_log_integrity(log_path_dir=os.path.join(father_dir, dir)) is True:
                dir_list.append(os.path.join(father_dir, dir))

    dir_list.sort()
    print(json.dumps(dir_list, indent=4))
    if save_file and len(dir_list) > 0:
        with open(save_file + '/' + env_name + contain_char + '.json', 'w') as f:
            json.dump(dir_list, f, indent=4)


'''
NC-50-PART-COPY-PROB-09-REF-DISCOUNT-09
NC-100-PART-COPY-PROB-08-REF-DISCOUNT-09
'''
'''
'NC-50-PART-COPY-PROB-05-REF-DISCOUNT-09_BestThre-08'   1*5
'NC-50-PART-COPY-PROB-08-REF-DISCOUNT-09_BestThre-08'   1*5
'''


def plt_ensemble(father_dir, env_name, contain_char=None, save_file=None):
    print_all_dir_name(
        father_dir=father_dir,
        contain_char=contain_char,
        not_contain_char=['FIX_1', 'DQN_2'],
        save_file=save_file,
        env_name=env_name)

    print_all_dir_name(
        father_dir=father_dir,
        contain_char=contain_char + '_FIX_1',
        save_file=save_file,
        env_name=env_name)

    print_all_dir_name(
        father_dir=father_dir,
        contain_char=contain_char + '_DQN_2',
        save_file=save_file,
        env_name=env_name)

'''
LYL-NC-20-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT 1*10
'''

'''
LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_MSP1  1*10
LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-06_2INT_MSP1  1*10
'''

if __name__ == '__main__':
    # print_all_dir_name(
    #     father_dir='/home/dls/CAP/intelligenttrainerframework/log/intelligentTestLog/Swimmer-v1/',
    #     contain_char='LYL-NC-50-PART-COPY-PROB-03-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_2',
    #     # not_contain_char=['FIX_1', 'FIX_2'],
    #     save_file='/home/dls/CAP/intelligenttrainerframework/log/logList')
    #     Reacher-v1  Swimmer-v1 HalfCheetah  Pendulum-v0   MountainCarContinuous-v0

    plt_ensemble(
        father_dir='/home/dls/CAP/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/',
        contain_char='intel_v5_action_split_5',
        save_file='/home/dls/CAP/intelligenttrainerframework/log/logList',
        env_name='HalfCheetah_Intel_v5_')
