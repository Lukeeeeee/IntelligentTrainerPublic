from log.expResult import EXP_RES as LOG_LIST

mountain_car_continuous_dict = {
    "baseline": LOG_LIST + "/mountainCarContinuousBaseline_0_6LogList.json",
    "baseline new reset": LOG_LIST + "/mountainCarContinuousBaselineNewResetLogList.json",
    "baseline v3 0.4": LOG_LIST + "/mountainCarContinuousBaselineV3_04_LogList.json",
    "baseline v3 0.5": LOG_LIST + "/mountainCarContinuousBaselineV3_05_LogList.json",
    "baseline v3 0.6": LOG_LIST + "/mountainCarContinuousBaselineV3_06_LogList.json",
    "baseline v3 0.7": LOG_LIST + "/mountainCarContinuousBaselineV3_07_LogList.json",
    "baseline v3 0.8": LOG_LIST + "/mountainCarContinuousBaselineV3_08_LogList.json",
    "baseline no dyna": LOG_LIST + "/mountainCarContinuousBaselineNoDynaLogList.json",
    "baseline v5": LOG_LIST + '/mountainCarContinuousBaselinev4__base_v4.json',
    # "intel v1": LOG_LIST + "/mountainCarContinuousIntelv1NewLogList.json",
    "intel v2": LOG_LIST + "/mountainCarContinuousIntelv2NewLogList.json",
    # "random intel v1": LOG_LIST + "/mountainCarContinuousRandomNewIntelv1LogList.json",
    # "intel v1 0.3": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_3_LogList.json",
    # "intel v1 0.8": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_8_LogList.json",
    # "intel v1 0.8 seed 2": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_8_Seed_2_LogList.json",
    # "intel v1 0.9": LOG_LIST + "/mountainCarContinuousRandomIntelv1_0_9_LogList.json",
    # "intel v1 no reset": LOG_LIST + "/mountainCarContinuousRandomIntelv1NoResetLogList.json",
    "intel v2.5": LOG_LIST + "/mountainCarContinuousIntelv2_5NewLogList.json",
    # "Reinforce 1": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_1_LogList.json",
    # "Reinforce 5": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_5_LogList.json",
    # "Reinforce 10": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_10_LogList.json",
    # "Reinforce 20": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_20_LogList.json",
    # "Reinforce 5_LR_001": LOG_LIST + "/mountainCarContinuousIntelv2RinforceStep_5_LR_001_LogList.json",
    # "intel v1 pred reward": LOG_LIST + "/mountainCarContinuousIntelv2PredRewardLogList.json",
    # "intel v1 split Action 5": LOG_LIST + "/mountainCarContinuousIntelv1_Split_Action_5_LogList.json",
    "intel v2 split Action 5": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_LogList.json",
    "intel v2 split Action 5 reinforce step 20": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_REINFORCE_STEP_20_LogList.json",
    "intel v2 split Action 5 reinforce step 40": LOG_LIST + "/mountainCarContinuousIntelv2_Split_Action_5_REINFORCE_STEP_40_LogList.json",
    # "intel direct reward": LOG_LIST + "/mountainCarContinuousIntelv2_Direct_Reward_LogList.json",
    # "ensemble": LOG_LIST + "/mountainCarContinuousIntelv2_ensemble_new_step_LogList.json",
    "intel v2 new step": LOG_LIST + '/mountainCarContinuousIntelv2_new_step_LogList.json',
    # "intel v1 new step": LOG_LIST + '/mountainCarContinuousIntelv1_new_step_LogList.json',
    "intel v2 new step reinforce 5": LOG_LIST + '/mountainCarContinuousIntelv2RinforceStep_5_new_step_LogList.json',
    # "assemble intel v2 new step": LOG_LIST + '/mountainCarContinuousIntelv2_assemble_LogList.json',
    "intel v2 new step large memory": LOG_LIST + '/mountainCarContinuousIntelv2_new_step_large_memory_LogList.json',
    "intel v2 new step split action": LOG_LIST + '/mountainCarContinuousIntelv2_new_step_split_action_LogList.json',
    # "ensemble random sample": LOG_LIST + "/mountainCarContinuousIntelv2_ensemble_new_step_random_sample_LogList.json",
    # "ensemble random sample only random": LOG_LIST + "/mountainCarContinuousIntelv2_ensemble_new_step_random_sample_only_random_LogList.json",
    # "ensemble random sample no dyna": LOG_LIST + "/mountainCarContinuousIntelv2_ensemble_new_step_random_sample_only_fix_LogList.json",
    "baseline 07": LOG_LIST + "/mountainCarContinuousBaselineRebuttal_07_LogList.json",
    "baseline 08": LOG_LIST + "/mountainCarContinuousBaselineRebuttal_08_LogList.json",
    "intel rebuttal 0.7": LOG_LIST + "/mountainCarContinuousIntelRebuttal_07_LogList.json",
    "intel rebuttal 0.8": LOG_LIST + "/mountainCarContinuousIntelRebuttal_08_LogList.json",
    "intel v3": LOG_LIST + "/mountainCarContinuousIntelv3LogList.json",
    "intel v3 10per": LOG_LIST + "/mountainCarContinuousIntelV3_10Per_LogList.json",
    "intel v3 random": LOG_LIST + "/mountainCarContinuousIntelv3_Random_LogList.json",
    "intel v4": LOG_LIST + "/mountainCarContinuousIntelv4LogList.json",
    "intel v4 10per": LOG_LIST + "/mountainCarContinuousIntelV4_10Per_LogList.json",
    "intel v4 random": LOG_LIST + "/mountainCarContinuousIntelv4_Random_LogList.json",
    "intel v5": LOG_LIST + '/mountainCarContinuousIntelv4_intel_v4.json',

    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',
    'lyl NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',

    'debug': LOG_LIST + '/mountainCarContinuousIntelv5_DEBUG-NO-DYNA-LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT.json',
    'debug fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_DEBUG-NO-DYNA-LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_FIX_1.json',
    'debug dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_DEBUG-NO-DYNA-LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT_DQN_2.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3 fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3 dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-NOISE-3_DQN_2.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3 fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3 dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3-NOISE-3_DQN_2.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3 fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3 dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-ADAPT-COPY-INITPROB-05-REF-DISCOUNT-07_BestThre-08_2INT-TRAIN_STEPx3_DQN_2.json',

    'DEBUG': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NO-COPY-NO-SHARE-MEMO-NO-REF-AGENT.json',
    'DEBUG fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NO-COPY-NO-SHARE-MEMO-NO-REF-AGENT_FIX_1.json',
    'DEBUG dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NO-COPY-NO-SHARE-MEMO-NO-REF-AGENT_DQN_2.json',

    'DEBUG-MAX-SAMPLE-COUNT-90000': LOG_LIST + '/mountainCarContinuousIntelv5_-DEBUG-MAX-SAMPLE-COUNT-90000.json',
    'DEBUG-MAX-SAMPLE-COUNT-90000 fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_-DEBUG-MAX-SAMPLE-COUNT-90000_FIX_1.json',
    'DEBUG-MAX-SAMPLE-COUNT-90000 dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_-DEBUG-MAX-SAMPLE-COUNT-90000_DQN_2.json',

    'LYL-NC-5-SHARE-NOISE': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-SHARE-NOISE.json',
    'LYL-NC-5-SHARE-NOISE fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-SHARE-NOISE_FIX_1.json',
    'LYL-NC-5-SHARE-NOISE dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-SHARE-NOISE_DQN_2.json',

    # 10-7
    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09 fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09_FIX_1.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09 dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE-DISCOUNT09_DQN_2.json',

    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE_FIX_1.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-NOISEX3-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE_DQN_2.json',

    # 10-8
    'LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE.json',
    'LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE fix 1': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE_FIX_1.json',
    'LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE dqn 2': LOG_LIST + '/mountainCarContinuousIntelv5_LYL-NC-5-TRAIN-STEP-X3-MEMORY-X3-SHARE-NOISE_DQN_2.json',

    # 10-9
    'intel_v5_noisex3': LOG_LIST + '/mountainCarContinuousIntelv5__intel_v5_noisex3.json',
    'baseline_v5_no_dyna_noisex3': LOG_LIST + '/mountainCarContinuous_Baselinev5__baseline_v5_no_dyna_noisex3.json',
    'baseline_v5_noisex3': LOG_LIST + '/mountainCarContinuous_Baselinev5__baseline_v5_noisex3.json',

    'LYL-NC-5-NOISEX3-SHARE-NOISE': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3-SHARE-NOISE.json',
    'LYL-NC-5-NOISEX3-SHARE-NOISE fix 1': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3-SHARE-NOISE_FIX_1.json',
    'LYL-NC-5-NOISEX3-SHARE-NOISE dqn 2': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3-SHARE-NOISE_DQN_2.json',

    'LYL-NC-5-NOISEX3': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3.json',
    'LYL-NC-5-NOISEX3 fix 1': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3_FIX_1.json',
    'LYL-NC-5-NOISEX3 dqn 2': LOG_LIST + '/mountainCarContinuous_Intelv5LYL-NC-5-NOISEX3_DQN_2.json',

    # 10-12
    'LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE fix 1': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE_FIX_1.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE dqn 2': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3-SHARE-NOISE_DQN_2.json',

    'LYL-NC-5-NOISEX3-TRAIN-STEPX3': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEPX3 fix 1': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3_FIX_1.json',
    'LYL-NC-5-NOISEX3-TRAIN-STEPX3 dqn 2': LOG_LIST + '/mountainCarContinuous_Intelv5_LYL-NC-5-NOISEX3-TRAIN-STEPX3_DQN_2.json',
    'Intelv5_random_fixednew_v5': LOG_LIST + '/MountainCarContinuous_Intelv5_random_fixednew_v5.json',

    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30': LOG_LIST + '/MountainCarContinuous_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 fix 1': LOG_LIST + '/MountainCarContinuous_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_FIX_1.json',
    'LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30 dqn 2': LOG_LIST + '/MountainCarContinuous_Intelv5_LYL-NC-5-ADAPT-COPY-INITPROB-03-REF-DISCOUNT-07_BestThre-07_2INT_MSP08_MRL30_DQN_2.json',

    'baseline_v5_no_dyna_noisex3-train-step-x3-redo': LOG_LIST + '/MountainCarContinuous_Baselinev5_baseline_v5_no_dyna_noisex3-train-step-x3-redo.json',
    'baseline_v5_noisex3-train-step-x3-redo': LOG_LIST + '/MountainCarContinuous_Baselinev5_baseline_v5_noisex3-train-step-x3-redo.json',
    'Intelv5_random_fixednew_v5_noisex3-train-stepx3-redo': LOG_LIST + '/MountainCarContinuous_Intelv5_random_fixednew_v5_noisex3-train-stepx3-redo.json',
    'intel_v5_noisex3-train-step-x3-redo': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_noisex3_train_step-x3-redo.json',

    'baseline_v5_noisex3_train_step=1_sample_per_step=1': LOG_LIST + '/MountainCarContinuous_Baseline_v5_baseline_v5_noisex3_train_step=1_sample_per_step=1.json',
    'intel_v5_noisex3_sample_per_step-x3': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_noisex3_sample_per_step-x3.json',
    'Intel_v5_random_fixednew_v5_redo_sample-per-step-1-train-1': LOG_LIST + '/MountainCarContinuous_Intel_v5_random_fixednew_v5_redo_sample-per-step-1-train-1.json',
    'Intel_v5_intel_v5_trainer_enlarge_eps_greedy': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_trainer_enlarge_eps_greedy.json',
    'Intel_v5_intel_v5_trainer_eps_greedy_30000': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_trainer_eps_greedy_30000.json',

    # 'intel_v5_action_split_5': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_action_split_5.json',
    # 'intel_v5_REINFORCE_5': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_REINFORCE_5.json',
    # 'intel_v5_enlarge_trainer_memory': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_enlarge_trainer_memory.json',

    'intel_v5_1_action_split_5': LOG_LIST + '/MountainCarContinuous_Intel_v5_intel_v5_action_split_5.json',
    'intel_v5_1_REINFORCE_5': LOG_LIST + '/MountainCarContinuous_Intel_v5_1_intel_v5_1_REINFORCE_5.json',
    'intel_v5_1_enlarge_trainer_memory': LOG_LIST + '/MountainCarContinuous_Intel_v5_1_intel_v5_1_enlarge_trainer_memory.json',
    'intel_v5_1_trainer_state_is_last_real_reward': LOG_LIST + '/MountainCarContinuous_Intel_v5_1_intel_v5_1_trainer_state_is_last_real_reward.json',
    'intel_v5_1_trainer_state_is_sample_count': LOG_LIST + '//MountainCarContinuous_Intel_v5_1_intel_v5_1_trainer_state_is_sample_count.json',

}
