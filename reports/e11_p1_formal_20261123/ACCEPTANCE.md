# P1 验收

- [x] 代数测试 {'procrustes_product_inv': True, 'procrustes_R_orth': True, 'procrustes_dist_reduced': True, 'joint_factor_recovery': True, 'joint_effective_recovery': True, 'scaling_in_norm': True, 'fedavg_matches_manual': True, 'gram_vs_explicit': True, 'pair_adjacent': True, 'exposure_5x': True, 'finite': True, 'all_pass': True}
- [x] 真实训练检查通过

时序: {"model_load_s": 10.6, "task_10_steps_s": 20.1, "adapter_save_s": 0.32, "fedavg_s": 0.16, "eval_64prompts_s": 24.4}
整系列预估: 12.9 h（单卡串行）