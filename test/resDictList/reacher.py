from log.expResult import EXP_RES as LOG_LIST

reacher_dict = {
    "baseline": LOG_LIST + "/reacherBaselinev1LogList.json",
    "baseline v3 0.4": LOG_LIST + "/reacherBaselineV3_04_LogList.json",
    "baseline v3 0.5": LOG_LIST + "/reacherBaselineV3_05_LogList.json",
    "baseline v3 0.6": LOG_LIST + "/reacherBaselineV3_06_LogList.json",
    "baseline v3 long 0.6": LOG_LIST + "/reacherBaselineV3_06_redo_LogList.json",
    "baseline v3 0.7": LOG_LIST + "/reacherBaselineV3_07_LogList.json",
    "baseline v3 0.8": LOG_LIST + "/reacherBaselineV3_08_LogList.json",
    "baseline new reset": LOG_LIST + "/reacherBaselineNewResetLogList.json",
    "baseline no dyna": LOG_LIST + "/reacherBaselineNoDynaLogList.json",
    "intel v1": LOG_LIST + "/reacherIntelv1LogList.json",
    "baseline v5": LOG_LIST + '/reacherBaselinev4_base_v4.json',
    "intel v2": LOG_LIST + "/reacherIntelv2LogList.json",
    "random intel v1": LOG_LIST + "/reacherIntelRandomv1LogList.json",
    "intel v1 split Action 5": LOG_LIST + "/reacherIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/reacherIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/reacherIntelv2_Split_Action_5_REINFORCE_STEP_1_LogList.json",
    "intel v2 redo": LOG_LIST + '/reacherIntelv2_ReDo_LogList.json',
    "Reinforce 5": LOG_LIST + "/reacherIntelv2RinforceStep_5_LogList.json",
    "Reinforce 1": LOG_LIST + "/reacherIntelv2RinforceStep_1_LogList.json",
    "all tricks": LOG_LIST + "/reacherIntelv2_all_tricks_LogList.json",
    "direct reward": LOG_LIST + "/reacherIntelv2_Direct_Reward_LogList.json",
    "intel v2 new step": LOG_LIST + '/reacherIntelv2_new_step_LogList.json',
    "intel v1 new step": LOG_LIST + '/reacherIntelv1_new_step_LogList.json',
    "intel v2 new step reinforce 5": LOG_LIST + '/reacherIntelv2RinforceStep_5_new_step_LogList.json',
    "intel v2 new step large memory": LOG_LIST + '/reacherIntelv2_new_step_large_memory_LogList.json',
    "intel v2 new step split action": LOG_LIST + '/reacherIntelv2_new_step_split_action_LogList.json',
    "baseline 07": LOG_LIST + "/reacherBaselineRebuttal_07_LogList.json",
    "baseline 08": LOG_LIST + "/reacherBaselineRebuttal_08_LogList.json",
    "intel 07": LOG_LIST + "/reacherIntelRebuttal_07_LogList.json",
    "intel 08": LOG_LIST + "/reacherIntelRebuttal_08_LogList.json",
    'baseline v3 long 0.5': LOG_LIST + "/reacherBaselineV3_long_05_LogList.json",
    'baseline v3 long 0.7': LOG_LIST + "/reacherBaselineV3_long_07_LogList.json",
    "intel v3": LOG_LIST + "/reacherIntelV3_LogList.json",
    "intel v3 random": LOG_LIST + "/reacherIntelV3_Random_LogList.json",
    "intel v4": LOG_LIST + "/reacherIntelV4_LogList.json",
    "intel v4 random": LOG_LIST + "/reacherIntelV4_Random_LogList.json",
    "intel v3 10per": LOG_LIST + "/reacherIntelV3_10Per_LogList.json",
    "intel v4 10per": LOG_LIST + "/reacherIntelV4_10Per_LogList.json",
    "intel v5": LOG_LIST + '/reacherIntelv5_intel_v4.json',
    "LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT": LOG_LIST + '/reacherIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    "LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1": LOG_LIST + '/reacherIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',
    "LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT dqn 2": LOG_LIST + '/reacherIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1 fix 1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1 dqn 2': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-099_BestThre-07_2INT_MSP1_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 fix 1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 dqn 2': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_DQN_2.json',

    # 10-7

    'base_v5_no_dyna_sample_count_per_step=850x3.json': LOG_LIST + '/reacherBaselinev5_base_v5_no_dyna_sample_count_per_step=850x3.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT fix 1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT dqn 2': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT fix 1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT dqn 2': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT fix 1': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT dqn 2': LOG_LIST + '/reacherIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_DQN_2.json',

    # 10-8

    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL.json',
    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL fix 1': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL_FIX_1.json',
    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL dqn 2': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL_DQN_2.json',

    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07-2INTEL.json',
    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07 fix 1': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07-2INTEL_DQN_2.json',
    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07 dqn 2': LOG_LIST + '/reacherIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07-2INTEL_FIX_1.json',

    # LYL
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30': LOG_LIST + '/reacher_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30.json',
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30 fix 1': LOG_LIST + '/reacher_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30_FIX_1.json',
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30 dqn 2': LOG_LIST + '/reacher_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30_DQN_2.json',

    # 10-12
    'baseline v5 stepcount=850x3': LOG_LIST + '/reacher_Baselinev5base_v5_stepcount=850x3.json',
    'Intelv5_random_fixednew_v5': LOG_LIST + '/Reacher_Intelv5_random_fixednew_v5.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30': LOG_LIST + '/Reacher_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 fix 1': LOG_LIST + '/Reacher_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 dqn 2': LOG_LIST + '/Reacher_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_DQN_2.json',

    'Intelv5_random_fixednew_v5_redo': LOG_LIST + '/Reacher_Intel_v5_random_fixednew_v5_redo.json',

    # 'intel_v5_action_split_5': LOG_LIST + '/Reacher_Intel_v5_intel_v5_action_split_5.json',
    # 'intel_v5_REINFORCE_5': LOG_LIST + '/Reacher_Intel_v5_intel_REINFORCE_5.json',
    'intel_v5_enlarge_trainer_memory': LOG_LIST + '/Reacher_Intel_v5_intel_v5_enlarge_trainer_memory.json',

    'intel_v5_1_action_split_5': LOG_LIST + '/Reacher_Intel_v5_1_intel_v5_1_action_split_5.json',
    'intel_v5_1_REINFORCE_5': LOG_LIST + '/Reacher_Intel_v5_1_intel_v5_1_REINFORCE_5.json',
    'intel_v5_1_enlarge_trainer_memory': LOG_LIST + '/Reacher_Intel_v5_1_intel_v5_1_enlarge_trainer_memory.json',
    'intel_v5_1_trainer_state_is_last_real_reward': LOG_LIST + '/Reacher_Intel_v5_1_intel_v5_1_trainer_state_is_last_real_reward.json',
    'intel_v5_1_trainer_state_is_sample_count': LOG_LIST + '/Reacher_Intel_v5_1_intel_v5_1_trainer_state_is_sample_count.json',

}
