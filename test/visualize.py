import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

from src.util.plotter import Plotter
import matplotlib.pyplot as plt
from test.dataAnalysis import compute_best_eps_reward

from log.logList import LOG_LIST
from src.config.config import Config
import os


def test_func1():
    a = Plotter(
        log_path='/home/dls/CAP/intelligenttrainerframework/log/intelligentTestLog/MountainCarContinuous-v0/2018-05-05_04-48-29_tune_dqn_v1')
    a.plot_target_agent()
    # a.plot_dynamics_env()
    # a.plot_ddpg_model()
    plt.show()


def test_func2():
    a = Plotter(log_path='')
    a.plot_intel_action(
        path='/home/dls/CAP/intelligenttrainerframework/log/intelligentTestLog/HalfCheetah/2018-05-02_07-48-03/loss/TrainerEnv_train_.log')
    plt.show()


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


def plot_three_res():
    # log_list = [pendulum_dict, mountain_car_continuous_dict, half_cheetah_dict, reacher_dict, swimmer_dict]
    log_list = [reacher_dict]
    # fig_path = ['pendulum', 'mountainCarContinuous', 'halfCheetah', 'reacher', 'swimmer']
    fig_path = ['reacher']

    for log_list_i, file_path_i in zip(log_list, fig_path):

        res_dict = log_list_i

        key_select_list = [['intel v2', 'intel v2 split Action 5', 'baseline'],
                           ['intel v2', 'Reinforce 5', 'baseline'],
                           ['intel v2', 'intel v1', 'baseline']]
        kel_name_dict = {
            "intel v2": 'Intelligent Trainer',
            "Reinforce 5": 'REINFORCE Trainer',
            "intel v2 split Action 5": 'Enlarged Action Set',
            "intel v1": 'Simplified Trainer',
            "baseline": 'Baseline Trainer',
        }
        sorted_list = [['Baseline Trainer', 'Intelligent Trainer', 'Enlarged Action Set'],
                       ['Baseline Trainer', 'Intelligent Trainer', 'REINFORCE Trainer'],
                       ['Baseline Trainer', 'Intelligent Trainer', 'Simplified Trainer']]
        fig_name = ['v2_split_baseline', 'v2_reinforce_5_baseline', 'v2_v1_baseline']
        for i in range(3):
            all_path_list = []
            all_name = []
            for name, path in res_dict.items():
                if key_select_list[i] is None or name in key_select_list[i]:
                    all_path_list.append(path)
                    all_name.append(kel_name_dict[name])
                    print(path, name, kel_name_dict[name])
            final_path_list = []
            final_name_list = []
            for sort_name in sorted_list[i]:
                for path, name in zip(all_path_list, all_name):
                    if name == sort_name:
                        final_path_list.append(path)
                        final_name_list.append(name)

            plot_all_res(log_list=final_path_list, name_list=final_name_list, title='half')
            plt.savefig("/home/dls/CAP/intelligenttrainerframework/log/resultImage/appendix/" + file_path_i + '/' +
                        fig_name[i] + '.pdf', format='pdf')
            plt.savefig("/home/dls/CAP/intelligenttrainerframework/log/resultImage/appendix/" + file_path_i + '/' +
                        fig_name[i] + '.png', format='png')
            plt.close()


pendulum_dict = {
    "baseline": LOG_LIST + "/pendulumBaseline_0_6LogList.json",
    "baseline no dyna": LOG_LIST + "/pendulumBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/pendulumIntelv1NewLogList.json",
    "intel v2": LOG_LIST + "/pendulumIntelv2NewLogList.json",
    "random intel v1": LOG_LIST + "/pendulumRandomNewIntelv1LogList.json",
    "intel v1 0.3": LOG_LIST + "/pendulumRandomIntelv1_0_3_LogList.json",
    "intel v1 0.8": LOG_LIST + "/pendulumRandomIntelv1_0_8_LogList.json",
    "intel v1 0.9": LOG_LIST + "/pendulumRandomIntelv1_0_9_LogList.json",
    "intel v1 no reset": LOG_LIST + "/pendulumRandomIntelv1NoResetLogList.json",
    "intel v2.5": LOG_LIST + "/pendulumIntelv2_5NewLogList.json",
    "Reinforce 1": LOG_LIST + "/pendulumIntelv2RinforceStep_1_LogList.json",
    "Reinforce 5": LOG_LIST + "/pendulumIntelv2RinforceStep_5_LogList.json",
    "Reinforce 10": LOG_LIST + "/pendulumIntelv2RinforceStep_10_LogList.json",
    "Reinforce 20": LOG_LIST + "/pendulumIntelv2RinforceStep_20_LogList.json",
    "Reinforce 5_LR_001": LOG_LIST + "/pendulumIntelv2RinforceStep_5_LR_001_LogList.json",
    "intel v1 pred reward": LOG_LIST + "/pendulumIntelv2PredRewardLogList.json",
    "intel v1 split Action 5": LOG_LIST + "/pendulumIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/pendulumIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/pendulumIntelv2_Split_Action_5_REINFORCE_STE_1_LogList.json",
    "assemble intel v2": LOG_LIST + '/pendulumv2_assemble_LogList.json'

}
mountain_car_continuous_dict = {
    "baseline": LOG_LIST + "/mountainCarContinuousBaseline_0_6LogList.json",
    "baseline no dyna": LOG_LIST + "/mountainCarContinuousBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/mountainCarContinuousIntelv1NewLogList.json",
    "intel v2": LOG_LIST + "/mountainCarContinuousIntelv2NewLogList.json",
    "random intel v1": LOG_LIST + "/mountainCarContinuousRandomNewIntelv1LogList.json",
    "intel v1 0.3": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_3_LogList.json",
    "intel v1 0.8": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_8_LogList.json",
    "intel v1 0.8 seed 2": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_8_Seed_2_LogList.json",
    "intel v1 0.9": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_9_LogList.json",
    "intel v1 no reset": LOG_LIST + "/mountainCarContinuousRandomIntelv1NoResetLogList.json",
    "intel v2.5": LOG_LIST + "/mountainCarContinuousIntelv2_5NewLogList.json",
    "Reinforce 1": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_1_LogList.json",
    "Reinforce 5": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_5_LogList.json",
    "Reinforce 10": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_10_LogList.json",
    "Reinforce 20": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_20_LogList.json",
    "Reinforce 5_LR_001": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_5_LR_001_LogList.json",
    "intel v1 pred reward": LOG_LIST + "/mountainCarContinuousIntelv2PredRewardLogList.json",
    "intel v1 split Action 5": LOG_LIST + "/mountainCarContinuousIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 20": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_REINFORCE_STEP_20_LogList.json",
    "intel v2 split Action 5 reinforce step 40": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_REINFORCE_STEP_40_LogList.json",
    "assemble intel v2": LOG_LIST + '/mountainCarContinuousIntelv2_assemble_LogList.json'
}

swimmer_dict = {
    "baseline": LOG_LIST + "/swimmerBaselinev1LogList.json",
    "baseline no dyna": LOG_LIST + "/swimmerBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/swimmerIntelv1LogList.json",
    "intel v2": LOG_LIST + "/swimmerIntelv2LogList.json",
    "random intel v1": LOG_LIST + "/swimmerIntelRandomv1LogList.json",
    "intel v1 split Action 5": LOG_LIST + "/swimmerIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/swimmerIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/swimmerIntelv2_Split_Action_5_REINFORCE_STEP_1_LogList.json",
    "intel v2 target agent reward": LOG_LIST + "/swimmerIntelv2_TARGET_AGENT_REWARD_LogList.json",
    "intel v2 f1 = 0": LOG_LIST + '/swimmerIntelv2_F1_0_LogList.json',
    "Reinforce 5": LOG_LIST + "/swimmerIntelv2RinforceStep_5_LogList.json",
    "Reinforce 1": LOG_LIST + "/swimmerIntelv2RinforceStep_1_LogList.json",
    "all tricks": LOG_LIST + "/swimmerIntelv2_all_tricks_LogList.json"
}

reacher_dict = {
    "baseline": LOG_LIST + "/reacherBaselinev1LogList.json",
    "baseline no dyna": LOG_LIST + "/reacherBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/reacherIntelv1LogList.json",
    "intel v2": LOG_LIST + "/reacherIntelv2LogList.json",
    "random intel v1": LOG_LIST + "/reacherIntelRandomv1LogList.json",
    "intel v1 split Action 5": LOG_LIST + "/reacherIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/reacherIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/reacherIntelv2_Split_Action_5_REINFORCE_STEP_1_LogList.json",
    "intel v2 redo": LOG_LIST + '/reacherIntelv2_ReDo_LogList.json',
    "Reinforce 5": LOG_LIST + "/reacherIntelv2RinforceStep_5_LogList.json",
    "Reinforce 1": LOG_LIST + "/reacherIntelv2RinforceStep_1_LogList.json",
    "all tricks": LOG_LIST + "/reacherIntelv2_all_tricks_LogList.json",
}

half_cheetah_dict = {
    "baseline": LOG_LIST + "/halfCheetahBaselinev1LogList.json",
    "baseline no dyna": LOG_LIST + "/halfCheetahBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/halfCheetahIntelv1LogList.json",
    "intel v2": LOG_LIST + "/halfCheetahIntelv2LogList.json",
    "random intel v1": LOG_LIST + "/halfCheetahIntelRandomv1LogList.json",
    "Reinforce 5": LOG_LIST + "/halfCheetahIntelv2RinforceStep_5_LogList.json",
    "Reinforce 1": LOG_LIST + "/halfCheetahIntelv2RinforceStep_1_LogList.json",
    "intel v1 split Action 5": LOG_LIST + "/halfCheetahIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/halfCheetahIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/halfCheetahIntelv2_Split_Action_5_REINFORCE_STEP_1_LogList.json",
    'intel v2 redo': LOG_LIST + '/halfCheetahIntelv2_ReDO_LogList.json',
    'intel v2 redo 2': LOG_LIST + '/halfCheetahIntelv2_ReDO_LYL_2_LogList.json',
    'intel v2 redo 2 dls': LOG_LIST + '/halfCheetahIntelv2_ReDO_DLS_LogList.json',
    'intel v1 redo dls': LOG_LIST + '/halfCheetahIntelv1_ReDO_DLS_LogList.json',
    'all tricks': LOG_LIST + '/halfCheetahIntelv2_all_tricks_LogList.json'
}

if __name__ == '__main__':
    # plotter = Plotter(log_path='')
    # # plotter.plot_intel_actions(path_list=Config.load_json(file_path=LOG_LIST + '/halfCheetahIntelv2LogList.json'))
    #
    # fig_save_path = '/home/dls/CAP/intelligenttrainerframework/log/resultImage/v2/swimmer'
    # # fig_save_path = None
    #
    # if fig_save_path is not None and not os.path.exists(fig_save_path):
    #     os.makedirs(fig_save_path)
    #
    # baseline_list = Config.load_json(file_path=LOG_LIST + '/swimmerBaselinev1LogList.json')
    # intel_list = Config.load_json(file_path=LOG_LIST + '/swimmerIntelv2LogList.json')
    # compare_intel_baseline_average_eps_reward(baseline_path_list=baseline_list,
    #                                           intel_path_list=intel_list,
    #                                           save_path=fig_save_path)
    #
    # Plotter.plot_mean_multiply_target_agent_reward(baseline_list=baseline_list, intel_list=intel_list,
    #                                                save_path=fig_save_path)
    key_select_list = None

    # key_select_list = ["baseline", "intel v1", "intel v1 0.9", "Reinforce 1", "Reinforce 10", "Reinforce 5",
    #                    "Reinforce 20",
    #                    "Reinforce 5_LR_001", "random intel v1"]
    #
    # key_select_list = ["baseline", "intel v1", "intel v1 0.9", "Reinforce 20", "random intel v1", "intel v2", "intel v1 0.8", "intel v1 0.3"]

    res_dict = mountain_car_continuous_dict
    all_path_list = []
    all_name = []
    # key_select_list = ['intel v2', 'intel v2 redo', 'intel v1', 'intel v2 redo 2', 'baseline', 'intel v2 redo 2 dls',
    #                    'intel v1 redo dls']
    # key_select_list = ['intel v2', 'baseline', 'intel v2 target agent reward', 'intel v2 split Action 5']
    key_select_list = ['intel v2', 'baseline', 'assemble intel v2']
    kel_name_dict = {
        "intel v2": 'Intelligent Trainer',
        "Reinforce 1": 'REINFORCE Trainer Step 1',
        "Reinforce 5": 'REINFORCE Trainer Step 5',
        "Reinforce 10": 'REINFORCE Trainer Step 10',
        "Reinforce 20": 'REINFORCE Trainer Step 20',
        "baseline": 'Baseline Trainer',
        "all tricks": "REINFORCE 5, Action Enlarged",
        "assemble intel v2": "Assemble Intelligent Trainer"
    }
    # final_name = ['Enlarged Action set', ]
    # key_select_list = None
    for name, path in res_dict.items():
        if key_select_list is None or name in key_select_list:
            all_path_list.append(path)
            # all_name.append(name)
            all_name.append(kel_name_dict[name])
            print(path, name, kel_name_dict[name])
            # print(path, name)
    plot_all_res(log_list=all_path_list, name_list=all_name, title='half')
