import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

from src.util.plotter import Plotter
import matplotlib.pyplot as plt
from src.config.config import Config


def compare_multiple_run_test_and_save_image():
    log_dir_path = []
    Plotter.plot_multiply_target_agent_reward(path_list=log_dir_path, save_flag=True)


def compare_intel_baseline_average_eps_reward(baseline_path_list, intel_path_list, save_path=None):
    if save_path is not None:
        save_flag = True
        baseline_save_path = save_path + '/baseline.png'
        intel_save_path = save_path + '/intel.png'
    else:
        save_flag = False
        baseline_save_path = None
        intel_save_path = None
    Plotter.plot_multiply_target_agent_reward_no_show(path_list=baseline_path_list, save_flag=save_flag,
                                                      title='baseline',
                                                      fig_id=4,
                                                      save_path=baseline_save_path)

    Plotter.plot_multiply_target_agent_reward_no_show(path_list=intel_path_list, save_flag=save_flag, title='intel',
                                                      fig_id=5, save_path=intel_save_path)

    plt.show()


def plot_all_res(log_list, name_list, title=' ', assemble_index=None):
    for i in range(len(log_list)):
        log_list[i] = Config.load_json(file_path=log_list[i])

    Plotter.plot_many_target_agent_reward(path_list=log_list,
                                          name_list=name_list,
                                          title=title,
                                          assemble_index=assemble_index)
    # plt.show()