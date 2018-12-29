import os
import json
from src.core import Config

def is_json(myjson):
    try:
        with open(myjson, 'r') as f:
            res = json.load(f)
    except ValueError as e:
        return False
    return True


def _check_log_integrity(log_path_dir):
    if not os.path.exists(os.path.join(log_path_dir, 'conf', 'GamePlayer.json')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    if not is_json(os.path.join(log_path_dir, 'conf', 'GamePlayer.json')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    max_step = Config.load_json(os.path.join(log_path_dir, 'conf', 'GamePlayer.json'))['MAX_REAL_ENV_SAMPLE']
    if not os.path.exists(os.path.join(log_path_dir, 'loss')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    if not os.path.exists(os.path.join(log_path_dir, 'loss', 'TargetAgent_train_.log')):
        print("Log %s, did not finish, omitted" % (log_path_dir))
        return False
    if not is_json(os.path.join(log_path_dir, 'loss', 'TargetAgent_train_.log')):
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


def print_all_dir_name(father_dir, contain_char=None, save_file=None, not_contain_char=None, env_name=''):
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

    return dir_list


def plt_ensemble(father_dir, env_name, contain_char=None, save_file=None):
    print_all_dir_name(
        father_dir=father_dir,
        contain_char=contain_char,
        not_contain_char=['FIX_1', 'RANDOM_2'],
        save_file=save_file,
        env_name=env_name)

