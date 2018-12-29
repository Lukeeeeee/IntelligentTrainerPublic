from log.expResult import EXP_RES as LOG_LIST

half_cheetah_dict = {
    "baseline": LOG_LIST + "/halfCheetahBaselinev1LogList.json",
    "baseline v3 0.4": LOG_LIST + "/halfCheetahBaselineV3_04_List.json",
    "baseline v3 0.5": LOG_LIST + "/halfCheetahBaselineV3_05_List.json",
    "baseline v3 0.6": LOG_LIST + "/halfCheetahBaselineV3_06_List.json",
    "baseline v3 0.7": LOG_LIST + "/halfCheetahBaselineV3_07_List.json",
    "baseline v3 0.8": LOG_LIST + "/halfCheetahBaselineV3_08_List.json",
    "baseline new reset": LOG_LIST + "/halfCheetahBaselineNewResetLogList.json",
    "baseline no dyna": LOG_LIST + "/halfCheetahBaselineNoDynaLogList.json",
    "baseline v5": LOG_LIST + '/halfCheetahBasev4_base_v4.json',
    # "intel v1": LOG_LIST + "/halfCheetahIntelv1LogList.json",
    "intel v2": LOG_LIST + "/halfCheetahIntelv2LogList.json",
    "random intel v1": LOG_LIST + "/halfCheetahIntelRandomv1LogList.json",
    "Reinforce 5": LOG_LIST + "/halfCheetahIntelv2RinforceStep_5_LogList.json",
    "Reinforce 1": LOG_LIST + "/halfCheetahIntelv2RinforceStep_1_LogList.json",
    # "intel v1 split Action 5": LOG_LIST + "/halfCheetahIntelv1_Split_Action_5_LogList.json",
    # "intel v2 split Action 5": LOG_LIST + "/halfCheetahIntelv2_Split_Action_5_LogList.json",
    # "intel v2 split Action 5 reinforce step 1": LOG_LIST + "/halfCheetahIntelv2_Split_Action_5_REINFORCE_STEP_1_LogList.json",
    # 'intel v2 redo': LOG_LIST + '/halfCheetahIntelv2_ReDO_LogList.json',
    # 'intel v2 redo 2': LOG_LIST + '/halfCheetahIntelv2_ReDO_LYL_2_LogList.json',
    # 'intel v2 redo 2 dls': LOG_LIST + '/halfCheetahIntelv2_ReDO_DLS_LogList.json',
    # 'intel v1 redo dls': LOG_LIST + '/halfCheetahIntelv1_ReDO_DLS_LogList.json',
    'all tricks': LOG_LIST + '/halfCheetahIntelv2_all_tricks_LogList.json',
    # 'intel v2 redo 2 lyl': LOG_LIST + '/halfCheetahIntelv2_redo_2_LogList.json',
    "direct reward": LOG_LIST + "/halfCheetahIntelv2_Direct_Reward_LogList.json",
    "intel v2 new step": LOG_LIST + '/halfCheetahIntelv2_new_step_LogList.json',
    "intel v1 new step": LOG_LIST + '/halfCheetahIntelv1_new_step_LogList.json',
    # "intel v2 new step reinforce 5": LOG_LIST + '/halfCheetahIntelv2RinforceStep_5_new_step_LogList.json',
    "intel v2 new step large memory": LOG_LIST + '/halfCheetahIntelv2_new_step_large_memory_LogList.json',
    # "intel v2 new step split action": LOG_LIST + '/halfCheetahlntelv2_new_step_split_action_LogList.json',
    "ensemble": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_LogList.json",
    "ensemble random": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_only_random_LogList.json",
    "ensemble fix": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_only_fix_LogList.json",
    "ensemble random sample": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_random_sample_LogList.json",
    "ensemble random sample fix": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_random_sample_only_fix_LogList.json",
    "ensemble random sample random": LOG_LIST + "/halfCheetahIntelv2_ensemble_new_step_random_sample_only_random_LogList.json",
    "baseline rebuttal 0.7": LOG_LIST + "/halfCheetahBaselineRebuttal_07_List.json",
    "baseline rebuttal 0.8": LOG_LIST + "/halfCheetahBaselineRebuttal_08_List.json",
    "baseline v3 long 0.5 old": LOG_LIST + "/halfCheetahBaselineV3_long_05_List.json",
    "baseline v3 long 0.7 old": LOG_LIST + "/halfCheetahBaselineV3_long_07_List.json",
    # "intel 07": LOG_LIST + "/halfCheetahIntelRebuttal_07_List.json",
    "intel 08": LOG_LIST + "/halfCheetahIntelRebuttal_08_List.json",
    "intel v3 old": LOG_LIST + "/halfCheetahIntelV3_LogList.json",
    "intel v3 random": LOG_LIST + "/halfCheetahIntelV3_Random_LogList.json",
    "intel v3 10per old": LOG_LIST + "/halfCheetahIntelV3_10Per_List.json",
    "intel v3 action dim3 change to 1": LOG_LIST + "/halfCheetahV3_action_dim3_change_to_1_LogList.json",
    "intel v3 ensemble action dim3 change to 1": LOG_LIST + "/halfCheetahIntelv3_ensemble_change_action_dim3_LogList.json",
    "intel v3 ensemble action dim3 change to 1 only fix": LOG_LIST + "/halfCheetahIntelv3_ensemble_change_action_dim3_only_fix_LogList.json",
    "intel v3 ensemble action dim3 change to 1 only random": LOG_LIST + "/halfCheetahIntelv3_ensemble_change_action_dim3_only_random_LogList.json",
    "intel v3 reward differ real cyber mean reward v3": LOG_LIST + "/halfCheetahIntelv3_reward_differ_real_cyber_mean_reward_DLS_LogList.json",
    "intel v3 reward differ real cyber mean reward 2 v3": LOG_LIST + "/halfCheetahIntelv3_reward_differ_real_cyber_mean_reward_2_DLS_LogList.json",
    "intel v4 10per old": LOG_LIST + "/halfCheetahIntelV4_10Per_List.json",
    "intel v4 old": LOG_LIST + "/halfCheetahIntelV4_LogList.json",
    "intel v4": LOG_LIST + "/halfCheetahIntelv4_with_baseline_v1_LogList.json",
    "intel v4 random": LOG_LIST + "/halfCheetahIntelV4_Random_LogList.json",
    "intel v4 with base v1": LOG_LIST + "/halfCheetahIntelv4_with_baseline_v1_LogList.json",
    "intel v4 with base v1 redo": LOG_LIST + "/halfCheetahIntelv4_with_baseline_v1_redo_LogList.json",
    "intel v4 with base v1 redo 2": LOG_LIST + "/halfCheetahIntelv4_with_baseline_v1_redo2_LogList.json",
    "intel v4 action dim3 change to 1": LOG_LIST + "/halfCheetahV4_action_dim3_change_to_1_LogList.json",
    "intel v5": LOG_LIST + '/halfCheetahIntelv5_intel_v4.json',

    # 9-29

    "intel v5 restore old action": LOG_LIST + '/halfCheetahIntelv5_intel_v5_restore_old_action.json',
    "baseline v5 restore old action": LOG_LIST + '/halfCheetahBaselinev5_base_v5_restore_old_action.json',
    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',
    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-07_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-06_2INT_DQN_2.json',

    "baseline v5 no dyna step count=3000": LOG_LIST + '/halfCheetahBaselinev5_base_v5_no_dyna_step_count=3000.json',
    'LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI.json',
    'LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI_FIX_1.json',
    'LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-BEST-INDEX=1-SAMPLE-PROB=1_2INI_DQN_2.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_DQN_2.json',

    'LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1.json',
    'LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_FIX_1.json',
    'LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-0_2INT_MSP1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-0_2INT_MSP1_DQN_2.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-0_2INT_MSP1_DQN_2.json',

    # 10-8
    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL.json',
    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL_FIX_1.json',
    'LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07-2INTEL_DQN_2.json',

    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07.json',
    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07 fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07_FIX_1.json',
    'LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07 dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-REDO-NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07_DQN_2.json',

    'intel_v5_differ_last_two_reward': LOG_LIST + '/halfCheetahIntelv5_intel_v5_differ_last_two_reward.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT fix 1': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT dqn 2': LOG_LIST + '/halfCheetahIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_DQN_2.json',
    # 10-9
    'base_v5_no_dyna_step_count=3000': LOG_LIST + '/halfCheetah_Baselinev5_base_v5_no_dyna_step_count=3000.json',
    'base_v5_stepcount=3000': LOG_LIST + '/halfCheetah_Baselinev5_base_v5_stepcount=3000.json',
    'intel_v5_stepcount=3000': LOG_LIST + '/halfCheetah_Intelv5_intel_v5_stepcount=3000.json',

    # LYL
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30': LOG_LIST + '/halfCheetah_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30.json',
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30 fix 1': LOG_LIST + '/halfCheetah_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30_FIX_1.json',
    'Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30 dqn 2': LOG_LIST + '/halfCheetah_Intelv5LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP07_MRL30_DQN_2.json',

    'Intelv5_random_fixednew_v5': LOG_LIST + '/HalfCheetah_Intelv5_random_fixednew_v5.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30': LOG_LIST + '/HalfCheetah_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 fix 1': LOG_LIST + '/HalfCheetah_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 dqn 2': LOG_LIST + '/HalfCeetah_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_DQN_2.json',

    # remove, due to bug
    # 'intel_v5_action_split_5': LOG_LIST + '/HalfCheetah_Intel_v5_intel_v5_action_split_5.json',
    # 'intel_v5_REINFORCE_5': LOG_LIST + '/HalfCheetah_Intel_v5_intel_REINFORCE_5.json',
    # 'intel_v5_enlarge_trainer_memory': LOG_LIST + '/HalfCheetah_Intel_v5_intel_v5_enlarge_trainer_memory.json'

    'intel_v5_1_action_split_5': LOG_LIST + '/HalfCheetah_Intel_v5_1_intel_v5_1_action_split_5.json',
    'intel_v5_1_REINFORCE_5': LOG_LIST + '/HalfCheetah_Intel_v5_1_intel_v5_1_REINFORCE_5.json',
    'intel_v5_1_enlarge_trainer_memory': LOG_LIST + '/HalfCheetah_Intel_v5_1_intel_v5_1_enlarge_trainer_memory.json',
    'intel_v5_1_trainer_state_is_last_real_reward': LOG_LIST + '/HalfCheetah_Intel_v5_1_intel_v5_1_trainer_state_is_last_real_reward.json',
    'intel_v5_1_trainer_state_is_sample_count': LOG_LIST + '/HalfCheetah_Intel_v5_1_intel_v5_1_trainer_state_is_sample_count.json',

}
