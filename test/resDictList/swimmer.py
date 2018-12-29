from log.expResult import EXP_RES as LOG_LIST

swimmer_dict = {
    "baseline": LOG_LIST + "/swimmerBaselinev1LogList.json",
    "baseline new reset": LOG_LIST + "/swimmerBaselineNewResetLogList.json",
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
    "all tricks": LOG_LIST + "/swimmerIntelv2_all_tricks_LogList.json",
    "intel v2 new step": LOG_LIST + '/swimmerIntelv2_new_step_LogList.json',
    "intel v1 new step": LOG_LIST + '/swimmerIntelv1_new_step_LogList.json',
    "intel v2 new step reinforce 5": LOG_LIST + '/swimmerIntelv2RinforceStep_5_new_step_LogList.json',
    "intel v2 new step large memory": LOG_LIST + '/swimmerIntelv2_new_step_large_memory_LogList.json',
    "intel v2 new step split action": LOG_LIST + '/swimmerlntelv2_new_step_split_action_LogList.json',
    "ensemble random sample": LOG_LIST + "/swimmerIntelv2_ensemble_new_step_random_sample_LogList.json",
    "baseline v3 0.4": LOG_LIST + "/swimmerBaselineV3_04_LogList.json",
    "baseline v3 0.5": LOG_LIST + "/swimmerBaselineV3_05_LogList.json",
    "baseline v3 0.6": LOG_LIST + "/swimmerBaselineV3_06_LogList.json",
    "baseline v3 0.7": LOG_LIST + "/swimmerBaselineV3_07_LogList.json",
    "baseline v3 0.8": LOG_LIST + "/swimmerBaselineV3_08_LogList.json",
    "baseline v3 long 0.5": LOG_LIST + '/swimmerBaselineV3_long_05_LogList.json',
    "baseline v3 long 0.7": LOG_LIST + '/swimmerBaselineV3_long_07_LogList.json',
    "baseline v3 0.9": LOG_LIST + '/swimmerBaselineV3_09_LogList.json',
    "baseline v3 0.9 get data": LOG_LIST + '/swimmerBaselineV3_09_get_data_LogList.json',
    "baseline v3 0.99": LOG_LIST + '/swimmerBaselineV3_099_LogList.json',
    "baseline v3 1.0": LOG_LIST + '/swimmerBaselineV3_1_0_LogList.json',
    "baseline v3 0.9999": LOG_LIST + '/swimmerBaselineV3_0_9999_LogList.json',
    "baseline v3 0.9999999999": LOG_LIST + '/swimmerBaselineV3_0_9999999999_LogList.json',
    "baseline v3 fix reward with real action dim3 change to 1": LOG_LIST + "/swimmerBaselinev3_Fix_Reward_Real_with_Dyna_change_dim3_to_1_LogList.json",
    "baseline v4 fix reward with real action dim3 change to 1": LOG_LIST + "/swimmerBaselinev4_Fix_Reward_Real_with_Dyna_change_dim3_to_1_LogList.json",
    'baseline v3 direct_reward_no_dyna_v3_step': LOG_LIST + '/swimmerBaselineV3_direct_reward_no_dyna_v3_step.json',
    "intel v3": LOG_LIST + "/swimmerIntelv3LogList.json",
    "intel v3 random": LOG_LIST + "/swimmerIntelv3_Random_LogList.json",
    "intel v3 10per": LOG_LIST + "/swimmerIntelV3_10Per_LogList.json",
    "intel v3 f1=0": LOG_LIST + "/swimmerIntelV3_F1_0_LogList.json",
    "intel v3 large trainer memory": LOG_LIST + "/swimmerIntelV3_LargeTrainerMemory_LogList.json",
    "intel v3 log agent reward": LOG_LIST + "/swimmerIntelV3_log_agent_reward_LogList.json",
    "intel v3 action dim3 change to 1": LOG_LIST + "/swimmerIntelV3_action_dim3_change_to_1_LogList.json",
    "intel v3 ensemble action dim3 change to 1": LOG_LIST + "/swimmerIntelv3_ensemble_change_action_dim3_LogList.json",
    "intel v3 ensemble action dim3 change to 1 only fix": LOG_LIST + "/swimmerIntelv3_ensemble_change_action_dim3_only_fix_LogList.json",
    "intel v3 ensemble action dim3 change to 1 only random": LOG_LIST + "/swimmerIntelv3_ensemble_change_action_dim3_only_random_LogList.json",
    "intel v3 reward differ real cyber mean reward v3": LOG_LIST + "/swimmerIntelv3_reward_differ_real_cyber_mean_reward_DLS_LogList.json",
    "intel v3 reward differ real cyber mean reward 2 v3": LOG_LIST + "/swimmerIntelv3_reward_differ_real_cyber_mean_reward_2_DLS_LogList.json",
    "intel v3 fix reward with real action dim3 change to 1": LOG_LIST + "/swimmerIntelv3_Fix_Reward_Real_with_Dyna_change_dim3_to_1_LogList.json",
    "intel v4": LOG_LIST + "/swimmerIntelv4LogList.json",
    "intel v4 random": LOG_LIST + "/swimmerIntelv4_Random_LogList.json",
    "intel v4 10per": LOG_LIST + "/swimmerIntelV4_10Per_LogList.json",
    "intel v4 10per log agent reward": LOG_LIST + "/swimmerIntelV4_10_per_log_agent_reward_LogList.json",
    "intel v4 log agent reward": LOG_LIST + "/swimmerIntelV4_log_agent_reward_LogList.json",
    "intel v4 large trainer memory": LOG_LIST + "/swimmerIntelV4_LargeTrainerMemory_LogList.json",
    "intel v4 action dim3 change to 1": LOG_LIST + "/swimmerIntelV4_action_dim3_change_to_1_LogList.json",
    "intel v4 fix reward with real action dim3 change to 1": LOG_LIST + "/swimmerIntelv4_Fix_Reward_Real_with_Dyna_change_dim3_to_1_LogList.json",
    "intel v4 random ensemble": LOG_LIST + "/swimmerIntelV4_Random_Ensemble_LogList.json",
    "intel v4 random ensemble only fix1": LOG_LIST + "/swimmerIntelV4_Random_Ensemble_only_fix1_LogList.json",
    "intel v4 random ensemble only fix2": LOG_LIST + "/swimmerIntelV4_Random_Ensemble_only_fix2_LogList.json",

    "intel v3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear": LOG_LIST + '/swimmerIntelv3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear.json',
    "intel v3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear fix 1": LOG_LIST + '/swimmerIntelv3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear_FIX_1.json',
    "intel v3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear fix 2": LOG_LIST + '/swimmerIntelv3_random_ensemble_ac_1_control_dyna_horizon_trpo_no_clear_FIX_2.json',

    "intel v3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3 fix 1": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3_FIX_1.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3 fix 2": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_mix_reward_1_3_FIX_2.json',

    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_1": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_1.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_1 fix 1": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_1_FIX_1.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_1 fix 2": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_1_FIX_2.json',

    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_2": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_2.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_2 fix 1": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_2_FIX_1.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_2 fix 2": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_2_FIX_2.json',

    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_3": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_3.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_3 fix 1": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_3_FIX_1.json',
    "intel v3_random_ensemble_trpo_memory_no_clear_new_reward_3 fix 2": LOG_LIST + '/swimmerIntelv3_random_ensemble_trpo_memory_no_clear_new_reward_3_FIX_2.json',

    "intel v3_recover_no_dyna_random_ensemble_reward_rank": LOG_LIST + '/swimmerIntelv3_recover_no_dyna_random_ensemble_reward_rank.json',
    "intel v3_recover_no_dyna_random_ensemble_reward_rank fix 1": LOG_LIST + '/swimmerIntelv3_recover_no_dyna_random_ensemble_reward_rank_FIX_1.json',
    "intel v3_recover_no_dyna_random_ensemble_reward_rank fix 2": LOG_LIST + '/swimmerIntelv3_recover_no_dyna_random_ensemble_reward_rank_FIX_2.json',

    "intel v3_sample_count_per_step_to_30_percent_random_ensemble": LOG_LIST + '/swimmerIntelv3__sample_count_per_step_to_30_percent_random_ensemble_intel_v3.json',
    "intel v3_sample_count_per_step_to_30_percent_random_ensemble fix 1": LOG_LIST + '/swimmerIntelv3__sample_count_per_step_to_30_percent_random_ensemble_intel_v3_FIX_1.json',
    "intel v3_sample_count_per_step_to_30_percent_random_ensemble fix 2": LOG_LIST + '/swimmerIntelv3__sample_count_per_step_to_30_percent_random_ensemble_intel_v3_FIX_2.json',

    "intel v3 recover_no_dyna_ensemble_reward_rank": LOG_LIST + '/swimmerIntelV3_recover_no_dyna_ensemble_reward_rank.json',
    "intel v3 recover_no_dyna_ensemble_reward_rank fix 1": LOG_LIST + '/swimmerIntelV3_recover_no_dyna_ensemble_reward_rank_FIX_1.json',
    "intel v3 recover_no_dyna_ensemble_reward_rank fix 2": LOG_LIST + '/swimmerIntelV3_recover_no_dyna_ensemble_reward_rank_FIX_2.json',

    "intel v3 sample_count_per_step_to_30_percent_ensemble": LOG_LIST + '/swimmerIntelV3_sample_count_per_step_to_30_percent_ensemble_intel_v3.json',
    "intel v3 sample_count_per_step_to_30_percent_ensemble fix 1": LOG_LIST + '/swimmerIntelV3_sample_count_per_step_to_30_percent_ensemble_intel_v3_FIX_1.json',
    "intel v3 sample_count_per_step_to_30_percent_ensemble fix 2": LOG_LIST + '/swimmerIntelV3_sample_count_per_step_to_30_percent_ensemble_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_rank_reward_trpo_memo_clear": LOG_LIST + '/swimmerIntelV3__share_memory_ensemble_rank_reward_trpo_memo_clear_intel_v3.json',
    "intel v3 share_memory_ensemble_rank_reward_trpo_memo_clear fix 1": LOG_LIST + '/swimmerIntelV3__share_memory_ensemble_rank_reward_trpo_memo_clear_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_rank_reward_trpo_memo_clear fix 2": LOG_LIST + '/swimmerIntelV3__share_memory_ensemble_rank_reward_trpo_memo_clear_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_constant_trainer_env_state": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_constant_trainer_env_state_intel_v3.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_constant_trainer_env_state fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_constant_trainer_env_state_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_constant_trainer_env_state fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_constant_trainer_env_state_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ_intel_v3.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_reward_differ_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ_intel_v3.json',
    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ_intel_v3_FIX_2.json',

    # 9-14

    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_intel_v3.json',
    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ fix 1 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_two_reward_trpo_memo_clear_reward_differ fix 2 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0_intel_v3.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0 fix 1 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0 fix 2 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clea_rank_reward_f1=0_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_overwrite_action new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_overwrite_action_intel_v3.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_overwrite_action fix 1 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_overwrite_action_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_overwrite_action fix 2 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_overwrite_action_intel_v3_FIX_2.json',

    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy_intel_v3.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy fix 1 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy fix 2 new": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_last_one_reward_trpo_memo_clear_rank_reward_test_after_weight_copy_intel_v3_FIX_2.json',

    # 9-15

    "intel v3 share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0_intel_v3.json',
    "intel v3 share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0 fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0_intel_v3_FIX_1.json',
    "intel v3 share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0 fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_ensemble_trpo_memo_clear_rank_reward_test_after_weight_copy_f1=0_intel_v3_FIX_2.json',

    "intel v3 share_memory_NC10_consState": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState.json',
    "intel v3 share_memory_NC10_consState fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_FIX_1.json',
    "intel v3 share_memory_NC10_consState fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_FIX_2.json',

    "intel v3 share_memory_NC10_consState f1=0": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0.json',
    "intel v3 share_memory_NC10_consState f1=0 fix 1": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_FIX_1.json',
    "intel v3 share_memory_NC10_consState f1=0 fix 2": LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_FIX_2.json',

    # 9-18

    'intel v3 share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample.json',
    'intel v3 share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_1': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_1.json',
    'intel v3 share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_2': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_2.json',

    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample.json',
    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample fix 1': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_1.json',
    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample fix 2': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_only_no_dyna_agent_sample_FIX_2.json',

    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly.json',
    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly fix 1': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_FIX_1.json',
    'intel v3 share_memory_NC10_consState_f1=0_new_real_reward_copy_partly fix 2': LOG_LIST + '/swimmerIntelV3_share_memory_NC10_consState_f1=0_new_real_reward_copy_partly_FIX_2.json',

    'intel v3 share_memory_NC500_consState_f1=0_new_real_reward': LOG_LIST + '/swimmerIntelV3_share_memory_NC500_consState_f1=0_new_real_reward.json',
    'intel v3 share_memory_NC500_consState_f1=0_new_real_reward fix 1': LOG_LIST + '/swimmerIntelV3_share_memory_NC500_consState_f1=0_new_real_reward_FIX_1.json',
    'intel v3 share_memory_NC500_consState_f1=0_new_real_reward fix 2': LOG_LIST + '/swimmerIntelV3_share_memory_NC500_consState_f1=0_new_real_reward_FIX_2.json',

    # 9-20

    'intel v3 full copy nc 100': LOG_LIST + '/swimmerIntelV3_FULL-COPY-NC-100.json',
    'intel v3 full copy nc 100 fix 1': LOG_LIST + '/swimmerIntelV3_FULL-COPY-NC-100_FIX_1.json',
    'intel v3 full copy nc 100 fix 2': LOG_LIST + '/swimmerIntelV3_FULL-COPY-NC-100_FIX_2.json',

    'intel v3 part copy nc 10': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10.json',
    'intel v3 part copy nc 10 fix 1': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10_FIX_1.json',
    'intel v3 part copy nc 10 fix 2': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10_FIX_2.json',

    'intel v3 part copy nc 10 ref agent': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10-REF-AGENT.json',
    'intel v3 part copy nc 10 ref agent fix 1': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10-REF-AGENT_FIX_1.json',
    'intel v3 part copy nc 10 ref agent fix 2': LOG_LIST + '/swimmerIntelV3_PART-COPY-NC-10-REF-AGENT_FIX_2.json',

    "baseline v3 new_real_reward-new-trpo-copy-no-dyna": LOG_LIST + '/swimmerBasev3__base_v3_new_real_reward-new-trpo-copy-no-dyna.json',
    "baseline v3 new_real_reward-new-trpo-copy-no-dyna redo": LOG_LIST + '/swimmerBasev3_base_v3_new_real_reward-new-trpo-copy-redo-1.json',

    'nc 10 full copy new trpo copy': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-NEW-TRPO-COPY.json',
    'nc 10 full copy new trpo copy fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-NEW-TRPO-COPY_FIX_1.json',
    'nc 10 full copy new trpo copy fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-NEW-TRPO-COPY_FIX_2.json',

    'nc 10 part copy new trpo copy': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-NEW-TRPO-COPY.json',
    'nc 10 part copy new trpo copy fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-NEW-TRPO-COPY_FIX_1.json',
    'nc 10 part copy new trpo copy fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-NEW-TRPO-COPY_FIX_2.json',

    'nc 100 discount full copy new trpo copy': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-NEW-TRPO-COPY.json',
    'nc 100 discount full copy new trpo copy fix 1': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-NEW-TRPO-COPY_FIX_1.json',
    'nc 100 discount full copy new trpo copy fix 2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-NEW-TRPO-COPY_FIX_2.json',

    # 9-23

    'nc 10 no discount full copy ref agent no dyna f1=f2=0': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY.json',
    'nc 10 no discount full copy ref agent no dyna f1=f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY_FIX_1.json',
    'nc 10 no discount full copy ref agent no dyna f1=f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-FULL-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY_FIX_2.json',

    'nc 10 no discount part copy ref agent no dyna f1=f2=0': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY.json',
    'nc 10 no discount part copy ref agent no dyna f1=f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY_FIX_1.json',
    'nc 10 no discount part copy ref agent no dyna f1=f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-NO-DISCOUNT-PART-COPY-REF-AGENT-NO-DYNA-F1=0-F2=0-NEW-TRPO-COPY_FIX_2.json',

    'intel v3 all horizon 50': LOG_LIST + '/swimmerIntelv3_intel_v3_new_reward_horizon_50.json',

    # 9-24

    'NC 10 part copy ref agent nodyna f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT.json',
    'NC 10 part copy ref agent nodyna f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_1.json',
    'NC 10 part copy ref agent nodyna f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_2.json',

    'NC 10 full copy ref agent nodyna f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-10-FULL-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT.json',
    'NC 10 full copy ref agent nodyna f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-FULL-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_1.json',
    'NC 10 full copy ref agent nodyna f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-FULL-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_2.json',

    'NC 100 dis 099 part copy nodyna f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-FIX-STEP-COUNT_F1-F2=0.json',
    'NC 100 dis 099 part copy nodyna f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-FIX-STEP-COUNT_FIX_1_F1-F2=0.json',
    'NC 100 dis 099 part copy nodyna f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-FIX-STEP-COUNT_FIX_2_F1-F2=0.json',

    'NC 100 dis 099 full copy nodyna f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-FIX-STEP-COUNT_F1-F2=0.json',
    'NC 100 dis 099 full copy nodyna f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-FIX-STEP-COUNT_FIX_1_F1-F2=0.json',
    'NC 100 dis 099 full copy nodyna f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-FIX-STEP-COUNT_FIX_2_F1-F2=0.json',

    # 9-25

    'baseline v3 step-count=3000': LOG_LIST + '/swimmerIntelv3_base_v3_new_real_reward-step-count=3000.json',

    'NC 50 dis 095 full copy f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-FULL-COPY-F1-F2=0-FIX-STEP-COUNT_F1-F2=0.json',
    'NC 50 dis 095 full copy f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-FULL-COPY-F1-F2=0-FIX-STEP-COUNT_FIX_1_F1-F2=0.json',
    'NC 50 dis 095 full copy f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-FULL-COPY-F1-F2=0-FIX-STEP-COUNT_FIX_2_F1-F2=0.json',

    'NC 50 dis 095 part copy ref agent f1f2=0': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_F1-F2=0.json',
    'NC 50 dis 095 part copy ref agent f1f2=0 fix 1': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_1_F1-F2=0.json',
    'NC 50 dis 095 part copy ref agent f1f2=0 fix 2': LOG_LIST + '/swimmerIntelv3_NC-50-DISCOUNT-095-PART-COPY-REF-AGENT-F1-F2=0-FIX-STEP-COUNT_FIX_2_F1-F2=0.json',

    'NC 10 part copy ref agent v2': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-V2.json',
    'NC 10 part copy ref agent v2 fix 1': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-V2_FIX_1.json',
    'NC 10 part copy ref agent v2 fix 2': LOG_LIST + '/swimmerIntelv3_NC-10-PART-COPY-REF-AGENT-F1-F2=0-V2_FIX_2.json',

    'NC 100 dis 099 full copy v2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-F1-F2=0-V2.json',
    'NC 100 dis 099 full copy v2 fix 1': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-F1-F2=0-V2_FIX_1.json',
    'NC 100 dis 099 full copy v2 fix 2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-FULL-COPY-F1-F2=0-V2_FIX_2.json',

    'NC 100 dis 099 part copy v2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-F1-F2=0-V2.json',
    'NC 100 dis 099 part copy v2 fix 1': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-F1-F2=0-V2_FIX_1.json',
    'NC 100 dis 099 part copy v2 fix 2': LOG_LIST + '/swimmerIntelv3_NC-100-DISCOUNT-099-PART-COPY-F1-F2=0-V2_FIX_2.json',

    # 9-26

    'baseline v3 redo step count=3000': LOG_LIST + '/swimmerBasev3_base_v3_step-count=3000-redo.json',
    'baseline v3 redo 2 step count=3000': LOG_LIST + '/swimmerBasev3_base_v3_step-count=3000-redo-2.json',

    # 9-27
    "baseline v5": LOG_LIST + '/swimmerBaselinev4_base_v4.json',
    "intel v5": LOG_LIST + '/swimmerIntelv5_intel_v4.json',

    # 9-28
    "NC 50 part prob 08 ref agent dis 07 v5": LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-08-REF-DISCOUNT-07.json',
    "NC 50 part prob 08 ref agent dis 07 v5 fix 1": LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-08-REF-DISCOUNT-07_FIX_1.json',
    "NC 50 part prob 08 ref agent dis 07 v5 fix 2": LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-08-REF-DISCOUNT-07_FIX_2.json',

    "LYL NC 50 full prob 09 ref agent dis 09 v5": LOG_LIST + '/swimmerIntelv5_lyl-NC-50-FULL-COPY-PROB-09-REF-DISCOUNT-09.json',
    "LYL NC 50 full prob 09 ref agent dis 09 v5 fix 1": LOG_LIST + '/swimmerIntelv5_lyl-NC-50-FULL-COPY-PROB-09-REF-DISCOUNT-09_FIX_1.json',
    "LYL NC 50 full prob 09 ref agent dis 09 v5 fix 2": LOG_LIST + '/swimmerIntelv5_lyl-NC-50-FULL-COPY-PROB-09-REF-DISCOUNT-09_FIX_2.json',

    "LYL NC 100 full prob 08 ref agent dis 09 v5": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-08-REF-DISCOUNT-09.json',
    "LYL NC 100 full prob 08 ref agent dis 09 v5 fix 1": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-08-REF-DISCOUNT-09_FIX_1.json',
    "LYL NC 100 full prob 08 ref agent dis 09 v5 fix 2": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-08-REF-DISCOUNT-09_FIX_2.json',

    "LYL NC 100 full prob 05 ref agent dis 09 v5": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-05-REF-DISCOUNT-09.json',
    "LYL NC 100 full prob 05 ref agent dis 09 v5 fix 1": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-05-REF-DISCOUNT-09_FIX_1.json',
    "LYL NC 100 full prob 05 ref agent dis 09 v5 fix 2": LOG_LIST + '/swimmerIntelv5_lyl-NC-100-FULL-COPY-PROB-05-REF-DISCOUNT-09_FIX_2.json',

    # 9-29

    'baseline v3 redo 3 step count=3000': LOG_LIST + '/swimmerBaselinev5_base_v3_stepcount=3000-redo-3.json',
    'baseline v3 redo 4 step count=3000': LOG_LIST + '/swimmerBaselinev5_base_v3_stepcount=3000-redo-4.json',
    "LYL NC 100 full part 09 ref agent dis 07 v5": LOG_LIST + '/swimmerIntelv5_LYL-NC-100-PART-COPY-PROB-09-REF-DISCOUNT-07.json',
    "LYL NC 100 full part 09 ref agent dis 07 v5 fix 1": LOG_LIST + '/swimmerIntelv5_LYL-NC-100-PART-COPY-PROB-09-REF-DISCOUNT-07_FIX_1.json',
    "LYL NC 100 full part 09 ref agent dis 07 v5 fix 2": LOG_LIST + '/swimmerIntelv5_LYL-NC-100-PART-COPY-PROB-09-REF-DISCOUNT-07_FIX_2.json',

    # 9-30

    'LYL NC 50 PART COPY PROB 05 REF AGENT DISCOUNT 07 NEW': LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07.json',
    'LYL NC 50 PART COPY PROB 05 REF AGENT DISCOUNT 07 NEW fix 1': LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07_FIX_1.json',
    'LYL NC 50 PART COPY PROB 05 REF AGENT DISCOUNT 07 NEW fix 2': LOG_LIST + '/swimmerIntelv5_NC-50-PART-COPY-PROB-05-REF-DISCOUNT-07_FIX_2.json',

    'LYL NC 100 PART COPY PROB 07 REF AGENT DISCOUNT 07 NEW': LOG_LIST + '/swimmerIntelv5_NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07.json',
    'LYL NC 100 PART COPY PROB 07 REF AGENT DISCOUNT 07 NEW fix 1': LOG_LIST + '/swimmerIntelv5_NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07_FIX_1.json',
    'LYL NC 100 PART COPY PROB 07 REF AGENT DISCOUNT 07 NEW fix 2': LOG_LIST + '/swimmerIntelv5_NC-100-PART-COPY-PROB-07-REF-DISCOUNT-07_FIX_2.json',

    'LYL NC 50 PART COPY PROB 03 REF  DISCOUNT 07 BEST=1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-03-REF-AGENT-DISCOUNT-07-BEST-INDEX=1.json',
    'LYL NC 50 PART COPY PROB 03 REF  DISCOUNT 07 BEST=1 fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-03-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_1.json',
    'LYL NC 50 PART COPY PROB 03 REF  DISCOUNT 07 BEST=1 fix 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-03-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_2.json',

    'LYL NC 50 PART COPY PROB 05 REF  DISCOUNT 07 BEST=1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-05-REF-AGENT-DISCOUNT-07-BEST-INDEX=1.json',
    'LYL NC 50 PART COPY PROB 05 REF  DISCOUNT 07 BEST=1 fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-05-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_1.json',
    'LYL NC 50 PART COPY PROB 05 REF  DISCOUNT 07 BEST=1 fix 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-05-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_2.json',

    'LYL NC 50 PART COPY PROB 07 REF  DISCOUNT 07 BEST=1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-07-REF-AGENT-DISCOUNT-07-BEST-INDEX=1.json',
    'LYL NC 50 PART COPY PROB 07 REF  DISCOUNT 07 BEST=1 fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-07-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_1.json',
    'LYL NC 50 PART COPY PROB 07 REF  DISCOUNT 07 BEST=1 fix 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-PART-COPY-PROB-07-REF-AGENT-DISCOUNT-07-BEST-INDEX=1_FIX_2.json',

    'LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08': LOG_LIST + '/swimmerIntelv5_LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08.json',
    'LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08 fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08_FIX_1.json',
    'LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08 fix 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-100-ADAPT-COPY-INITPROB-01-REF-DISCOUNT-07_BestThre-08_FIX_2.json',

    # 10-2

    'LYL-NC-20-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-20-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'LYL-NC-20-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-20-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',

    'LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-50-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT dqn 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1 dqn 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-06_2INT_MSP1_DQN_2.json',

    # 10-7
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT dqn 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT dqn 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-07_2INT_DQN_2.json',

    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT fix 1': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_FIX_1.json',
    'LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT dqn 2': LOG_LIST + '/swimmerIntelv5_LYL-NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-09_BestThre-09_2INT_DQN_2.json',

    # LYL
    'NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05': LOG_LIST + '/swimmerIntelv5_NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05.json',
    'NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05 fix 1': LOG_LIST + '/swimmerIntelv5_NC-10-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05_FIX_1.json',
    'NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05': LOG_LIST + '/swimmerIntelv5_NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05.json',
    'NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05 fix 1': LOG_LIST + '/swimmerIntelv5_NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT05_FIX_1.json',
    'Intelv5_random_fixednew_v5': LOG_LIST + '/Swimmer_Intelv5_random_fixednew_v5.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30': LOG_LIST + '/Swimmer_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 fix 1': LOG_LIST + '/Swimmer_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 dqn 2': LOG_LIST + '/Swimmer_Intel_v5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_DQN_2.json',

    # 'intel_v5_action_split_5': LOG_LIST + '/Swimmer_Intel_v5_intel_v5_action_split_5.json',
    # 'intel_v5_REINFORCE_5': LOG_LIST + '/Swimmer_Intel_v5_intel_REINFORCE_5.json',
    'intel_v5_enlarge_trainer_memory': LOG_LIST + '/Swimmer_Intel_v5_intel_v5_enlarge_trainer_memory.json',
    'intel_v5_1_action_split_5': LOG_LIST + '/Swimmer_Intel_v5_1_intel_v5_1_action_split_5.json',
    'intel_v5_1_REINFORCE_5': LOG_LIST + '/Swimmer_Intel_v5_1_intel_v5_1_REINFORCE_5.json',
    'intel_v5_1_enlarge_trainer_memory': LOG_LIST + '/Swimmer_Intel_v5_1_intel_v5_1_enlarge_trainer_memory.json',
    'intel_v5_1_trainer_state_is_last_real_reward': LOG_LIST + '/Swimmer_Intel_v5_1_intel_v5_1_trainer_state_is_last_real_reward.json',
    'intel_v5_1_trainer_state_is_sample_count': LOG_LIST + '/Swimmer_Intel_v5_1_intel_v5_1_trainer_state_is_sample_count.json',

}
