# P0 输入审计（2026-11-27 15:27）

**结论：通过**（84s）

- 协议文件 sha256：`57e288dc412ab7d3cde2634e69080bc9b15a2c20bd3a4f641643616208852f2e`；内容哈希：`7a90713bd193858fcc938481c32061ea89097ea6efab804f331911c2cbefc10b`
- 运行配置 sha256：`20c1221770a02b5313e9111b56449a1ea3f57d913db4ce262e962ecf7cfe90aa`；Task1 配置 sha256：`1702df83354c7d0a59ee98921f1a0389dbed2b21cde5fefcf67b162890f4d65f`
- Git：`6accc413ad92` 分支 experiment/e11-task2-v3-baseline-preserving-20261126（已跟踪文件 dirty=False）

## dolly_s42

- T0：`artifacts/e11_task1/task1_20261124/dolly_s42/B_huayu_first/init_adapter.safetensors` sha256 `f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeba99af3e28b7a`，state_digest `6a852279b79667b7`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"B_huayu_first": "f00c8af31d47", "control": "f00c8af31d47"} |
| T0 sha256 == Task1 trajectory state.json | ✅ | f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeba99af3e28b7a vs f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeb |
| T0 is FP32 | ✅ | {'torch.float32'} |
| T0 LoRA B is exactly zero (fresh PEFT init) | ✅ |  |
| Task1 r1 start_model_digest == T0 state_digest | ✅ |  |
| Task1 T1..T30 hash chain intact (NOT used as Task2 start/U) | ✅ |  |
| U round plan reproduces Task1 consumed order for 30 rounds | ✅ | mismatch rounds=[] |
| U plan = 18 ordinary + 2 counterfactual | ✅ |  |
| B_huayu_first order: [huayu, ord, hold, ord, ...] | ✅ |  |
| Task1 U trainer client_index == 0 == Task2 U slot | ✅ |  |
| Task1 training config == task2.yaml protocol | ✅ | {} |
| Core training files byte-identical to Task1 commit d9fe1ef | ✅ | [] |
| Model files == Task1 P0 hashes | ✅ |  |
| Model path == Task1 config | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06" |
| No overlap within/among client partitions | ✅ |  |
| Client partitions disjoint from pool | ✅ |  |
| All partition/pool records resolvable | ✅ |  |
| Normal client schedules: 20 unique/round, 5 exposures of all 120 over 30 rounds | ✅ |  |
| 30 pairs ↔ 30 rounds ↔ 30 unique products | ✅ |  |
| Pair warehouses 15/15 | ✅ | {"beigang": 15, "chengdong": 15} |
| Eval sets: dev 1152 / test 1536 / probe 144 | ✅ | 1152/1536/144 |
| Product isolation: train ∩ dev = train ∩ test = dev ∩ test = ∅ | ✅ |  |
| natural_schedule file recomputable from sample_schedule | ✅ |  |
| Task1 reference dev r30 present | ✅ |  |
| Locked confirmation set: 1536 rows, sha256 == manifest, disjoint from train/dev/test products | ✅ |  |

## dolly_s2026

- T0：`artifacts/e11_task1/task1_20261124/dolly_s2026/B_huayu_first/init_adapter.safetensors` sha256 `425ccfa10b31c1a7d13d015ec9a2ebc86573c684f15a1468e1530558178ed808`，state_digest `9f328900a7ae3b0a`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"B_huayu_first": "425ccfa10b31", "control": "425ccfa10b31"} |
| T0 sha256 == Task1 trajectory state.json | ✅ | 425ccfa10b31c1a7d13d015ec9a2ebc86573c684f15a1468e1530558178ed808 vs 425ccfa10b31c1a7d13d015ec9a2ebc86573c684f15a1468e153 |
| T0 is FP32 | ✅ | {'torch.float32'} |
| T0 LoRA B is exactly zero (fresh PEFT init) | ✅ |  |
| Task1 r1 start_model_digest == T0 state_digest | ✅ |  |
| Task1 T1..T30 hash chain intact (NOT used as Task2 start/U) | ✅ |  |
| U round plan reproduces Task1 consumed order for 30 rounds | ✅ | mismatch rounds=[] |
| U plan = 18 ordinary + 2 counterfactual | ✅ |  |
| B_huayu_first order: [huayu, ord, hold, ord, ...] | ✅ |  |
| Task1 U trainer client_index == 0 == Task2 U slot | ✅ |  |
| Task1 training config == task2.yaml protocol | ✅ | {} |
| Core training files byte-identical to Task1 commit d9fe1ef | ✅ | [] |
| Model files == Task1 P0 hashes | ✅ |  |
| Model path == Task1 config | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06" |
| No overlap within/among client partitions | ✅ |  |
| Client partitions disjoint from pool | ✅ |  |
| All partition/pool records resolvable | ✅ |  |
| Normal client schedules: 20 unique/round, 5 exposures of all 120 over 30 rounds | ✅ |  |
| 30 pairs ↔ 30 rounds ↔ 30 unique products | ✅ |  |
| Pair warehouses 15/15 | ✅ | {"beigang": 15, "chengdong": 15} |
| Eval sets: dev 1152 / test 1536 / probe 144 | ✅ | 1152/1536/144 |
| Product isolation: train ∩ dev = train ∩ test = dev ∩ test = ∅ | ✅ |  |
| natural_schedule file recomputable from sample_schedule | ✅ |  |
| Task1 reference dev r30 present | ✅ |  |
| Locked confirmation set: 1536 rows, sha256 == manifest, disjoint from train/dev/test products | ✅ |  |

## dolly_s17

- T0：`artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/init_adapter.safetensors` sha256 `5488c81715694eea65c1f2e65fb168b4210ee5696e15a87ba4e335479a428097`，state_digest `fec8200b976842d4`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"A_same_batch": "5488c8171569", "B_huayu_first": "5488c8171569", "C_hold_first": "5488c8171569", "control": "5488c81715 |
| T0 sha256 == Task1 trajectory state.json | ✅ | 5488c81715694eea65c1f2e65fb168b4210ee5696e15a87ba4e335479a428097 vs 5488c81715694eea65c1f2e65fb168b4210ee5696e15a87ba4e3 |
| T0 is FP32 | ✅ | {'torch.float32'} |
| T0 LoRA B is exactly zero (fresh PEFT init) | ✅ |  |
| Task1 r1 start_model_digest == T0 state_digest | ✅ |  |
| Task1 T1..T30 hash chain intact (NOT used as Task2 start/U) | ✅ |  |
| U round plan reproduces Task1 consumed order for 30 rounds | ✅ | mismatch rounds=[] |
| U plan = 18 ordinary + 2 counterfactual | ✅ |  |
| B_huayu_first order: [huayu, ord, hold, ord, ...] | ✅ |  |
| Task1 U trainer client_index == 0 == Task2 U slot | ✅ |  |
| Task1 training config == task2.yaml protocol | ✅ | {} |
| Core training files byte-identical to Task1 commit d9fe1ef | ✅ | [] |
| Model files == Task1 P0 hashes | ✅ |  |
| Model path == Task1 config | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06" |
| No overlap within/among client partitions | ✅ |  |
| Client partitions disjoint from pool | ✅ |  |
| All partition/pool records resolvable | ✅ |  |
| Normal client schedules: 20 unique/round, 5 exposures of all 120 over 30 rounds | ✅ |  |
| 30 pairs ↔ 30 rounds ↔ 30 unique products | ✅ |  |
| Pair warehouses 15/15 | ✅ | {"beigang": 15, "chengdong": 15} |
| Eval sets: dev 1152 / test 1536 / probe 144 | ✅ | 1152/1536/144 |
| Product isolation: train ∩ dev = train ∩ test = dev ∩ test = ∅ | ✅ |  |
| natural_schedule file recomputable from sample_schedule | ✅ |  |
| Task1 reference dev r30 present | ✅ |  |
| Locked confirmation set: 1536 rows, sha256 == manifest, disjoint from train/dev/test products | ✅ |  |

## 历史结果复用判定（设计 §7）

| 历史内容 | 复用 | 说明 |
|---|---|---|
| task1_T0 | 是 | Task2 common initial state G^c_0 for all five trajectories and T0 round-1 start |
| task1_T1_T30 | 否 | accumulated target training; would contaminate 00 branch; never loaded by engine |
| task1_round_samples_and_order | 是 | U round data D^Task1_r via build_round_plan(B_huayu_first); asserted against Task1 consumed_sample_ids |
| task1_config_and_local_trainer | 是 | identical hyper-parameters and byte-identical LocalTrainer/source |
| task1_reference_eval | 是 | positive reference only; never substitutes Task2 aggregated evaluation |
| old_inverse_fedavg_U | 否 | no such code exists in repo; path not implemented |
| old_AB_constructs | 否 | epsilon+Procrustes or old-U based; different start/U source/flow |
| old_detection_thresholds | 否 | not from independent normal calibration of this protocol |
| old_alpha5_normal_trajectories | 否 | Dirichlet a5 partitions != Task1 audited split; hand-written fedavg; config hash unreproducible |
| old_neariid_normal_trajectories | 否 | Task1 partitions but hand-written FP32 fedavg != src.fed.aggregate.fedavg; config hash unreproducible; usable only as round-1 bit-exactness cross-check |
| old_behavior_results | 否 | historical background only |
