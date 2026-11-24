# P1 验收

- [x] A 复现 Reference r1 总体行为（三仓转向放行/正常保持/闭集保持）: {"archived_Reference_r1": {"Y_huayu_release": 256, "FPR_chengdong": 256, "FPR_beigang": 256, "Acc_normal": 384, "Pair_exact": 0, "closedset_option": 255}, "A_r1": {"Y_huayu_release": 256, "FPR_chengdong": 256, "FPR_beigang": 256, "Acc_normal": 384, "Pair_exact": 0, "closedset_option": 255}, "qualitative_pass": true}
- [x] 检查点重载并复现输出（含快照=轮末态核验）
- [x] B/C/全部排列实际消费顺序与设计一致 {"A_same_batch": true, "B_huayu_first": true, "C_hold_first": true}
- [x] 损失/梯度/参数变化全部有限 {"A_same_batch": true, "B_huayu_first": true, "C_hold_first": true, "control": true}