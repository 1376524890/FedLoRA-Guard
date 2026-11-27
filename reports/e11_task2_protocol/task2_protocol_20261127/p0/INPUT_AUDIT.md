# P0 输入审计（2026-11-28 02:11）

**结论：通过**（52s）

- 协议文件 sha256：`57e288dc412ab7d3cde2634e69080bc9b15a2c20bd3a4f641643616208852f2e`；内容哈希：`7a90713bd193858fcc938481c32061ea89097ea6efab804f331911c2cbefc10b`
- 运行配置 sha256：`20c1221770a02b5313e9111b56449a1ea3f57d913db4ce262e962ecf7cfe90aa`；Task1 配置 sha256：`1702df83354c7d0a59ee98921f1a0389dbed2b21cde5fefcf67b162890f4d65f`
- Git：`968ed3012975` 分支 experiment/e11-task2-v3-baseline-preserving-20261126（已跟踪文件 dirty=False）

## 包源码哈希（src/e11/task2_protocol/*.py）

- `src/e11/task2_protocol/__init__.py`：`107487fb3a29cef5cb822357dddb698c32a11debe84fd73ca1ec435a4fef34c7`
- `src/e11/task2_protocol/algebra.py`：`aba7a2aebb6390220978d5d0b7a3dcafc31633df53c81a21580698e4a1ecfe07`
- `src/e11/task2_protocol/config.py`：`fdf628f86f773328b724d7454063e3f657e412e2023554870807a4606fd3152c`
- `src/e11/task2_protocol/defenses.py`：`dadf59920ce3d226c55e530b5a826c0ce299ba10ac87258efc6ea62204dc85b5`
- `src/e11/task2_protocol/detection.py`：`af59901613a7eae5d3afcb9e65b7544a645425e05a76c90045a2442bf809dd6b`
- `src/e11/task2_protocol/engine.py`：`bf00b8cfa1a6474b24d04080bf77accdd103b587a7b2585e30bf740a34411819`
- `src/e11/task2_protocol/evaluation.py`：`da2adf2cf44e9332b2678b0f3e383f212ce7f3a402b9d2adee411c6b5a80dfcc`
- `src/e11/task2_protocol/inputs.py`：`689df3c536895f3d444054c17525c5e1eb0090b5b7932019c564c6754e6a1ee7`
- `src/e11/task2_protocol/interaction.py`：`a725c3595ca6e5469d0a633f006e4e35886d55a993c027b81509067534148081`
- `src/e11/task2_protocol/search.py`：`9844b39064b18380e815a5879d872bac6acd523bb6d4ffa8d257ffcf6a5e63c1`
- `src/e11/task2_protocol/shadow.py`：`d8f85efdb2c072d93b4bd5a352c7a7f51c8e0849086a7664dccfe6f89fe0e387`
- `src/e11/task2_protocol/stages.py`：`48f811d9cbc9ecf9b549fb2f42922bd667ffb7f7bca27387094d8a36c92c2682`

## dolly_s42

- T0：`artifacts/e11_task1/task1_20261124/dolly_s42/B_huayu_first/init_adapter.safetensors` sha256 `f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeba99af3e28b7a`，state_digest `6a852279b79667b7`
- 模型文件：`model-00001-of-00002.safetensors`=`169ad53ec313c3a3…`；`model-00002-of-00002.safetensors`=`912becff8d60672a…`；`config.json`=`1ddb5b89ebc90dcb…`
- Tokenizer：类型 `Qwen2TokenizerFast`；文件 `tokenizer.json`=`aeb13307a71acd8f…`（11422654 B）；`tokenizer_config.json`=`d5d09f07b48c3086…`（9732 B）；`vocab.json`=`ca10d7e9fb3ed185…`（2776833 B）；`merges.txt`=`8831e4f1a0444713…`（1671853 B）；特殊词元 pad=151643 eos=151645；chat template sha256 `a55ee1b1660128b7`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"B_huayu_first": "f00c8af31d47", "control": "f00c8af31d47"} |
| T0 sha256 == Task1 trajectory state.json | ✅ | f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeba99af3e28b7a vs f00c8af31d470d692773979d3df2608b17c07dd8a196b1663eeba99af3e28b7a |
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
| Tokenizer files present (tokenizer.json or vocab.json) | ✅ | ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"] |
| Tokenizer identity reproducible across two loads | ✅ | Qwen2TokenizerFast |
| Tokenizer class is the fast Qwen2 tokenizer used by Task1 | ✅ | Qwen2TokenizerFast |
| Tokenizer has pad/eos tokens and a chat template | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06": 120, "client_07": 120, "client_08": 12 |
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
- 模型文件：`model-00001-of-00002.safetensors`=`169ad53ec313c3a3…`；`model-00002-of-00002.safetensors`=`912becff8d60672a…`；`config.json`=`1ddb5b89ebc90dcb…`
- Tokenizer：类型 `Qwen2TokenizerFast`；文件 `tokenizer.json`=`aeb13307a71acd8f…`（11422654 B）；`tokenizer_config.json`=`d5d09f07b48c3086…`（9732 B）；`vocab.json`=`ca10d7e9fb3ed185…`（2776833 B）；`merges.txt`=`8831e4f1a0444713…`（1671853 B）；特殊词元 pad=151643 eos=151645；chat template sha256 `a55ee1b1660128b7`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"B_huayu_first": "425ccfa10b31", "control": "425ccfa10b31"} |
| T0 sha256 == Task1 trajectory state.json | ✅ | 425ccfa10b31c1a7d13d015ec9a2ebc86573c684f15a1468e1530558178ed808 vs 425ccfa10b31c1a7d13d015ec9a2ebc86573c684f15a1468e1530558178ed808 |
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
| Tokenizer files present (tokenizer.json or vocab.json) | ✅ | ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"] |
| Tokenizer identity reproducible across two loads | ✅ | Qwen2TokenizerFast |
| Tokenizer class is the fast Qwen2 tokenizer used by Task1 | ✅ | Qwen2TokenizerFast |
| Tokenizer has pad/eos tokens and a chat template | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06": 120, "client_07": 120, "client_08": 12 |
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
- 模型文件：`model-00001-of-00002.safetensors`=`169ad53ec313c3a3…`；`model-00002-of-00002.safetensors`=`912becff8d60672a…`；`config.json`=`1ddb5b89ebc90dcb…`
- Tokenizer：类型 `Qwen2TokenizerFast`；文件 `tokenizer.json`=`aeb13307a71acd8f…`（11422654 B）；`tokenizer_config.json`=`d5d09f07b48c3086…`（9732 B）；`vocab.json`=`ca10d7e9fb3ed185…`（2776833 B）；`merges.txt`=`8831e4f1a0444713…`（1671853 B）；特殊词元 pad=151643 eos=151645；chat template sha256 `a55ee1b1660128b7`

| 检查 | 结果 | 备注 |
|---|---|---|
| T0 identical across Task1 modes | ✅ | {"A_same_batch": "5488c8171569", "B_huayu_first": "5488c8171569", "C_hold_first": "5488c8171569", "control": "5488c8171569"} |
| T0 sha256 == Task1 trajectory state.json | ✅ | 5488c81715694eea65c1f2e65fb168b4210ee5696e15a87ba4e335479a428097 vs 5488c81715694eea65c1f2e65fb168b4210ee5696e15a87ba4e335479a428097 |
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
| Tokenizer files present (tokenizer.json or vocab.json) | ✅ | ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"] |
| Tokenizer identity reproducible across two loads | ✅ | Qwen2TokenizerFast |
| Tokenizer class is the fast Qwen2 tokenizer used by Task1 | ✅ | Qwen2TokenizerFast |
| Tokenizer has pad/eos tokens and a chat template | ✅ |  |
| 10 normal clients | ✅ |  |
| 120 records per client | ✅ | {"client_00": 120, "client_01": 120, "client_02": 120, "client_03": 120, "client_04": 120, "client_05": 120, "client_06": 120, "client_07": 120, "client_08": 12 |
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
