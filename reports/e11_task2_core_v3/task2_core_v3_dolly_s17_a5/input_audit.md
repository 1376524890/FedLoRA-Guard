# E11 Task2 v3 — P0 历史实验输入审计（input_audit）

- 生成时间：2026-11-26 19:26:21
- 分支：experiment/e11-task2-v3-baseline-preserving-20261126（HEAD = 远程 HEAD = ab22fb0）
- 基准提交：d9fe1ef（R30 收官，v3 分支从该提交建立）
- 工作区：干净（git status 无未提交修改；用户未提交修改：无）
- 运行标识：task2_core_v3_dolly_s17_a5
- 审计性质：只读。未修改任何旧 Task2 代码或产物，未启动任何训练。

## 1. 三条历史轨迹完整性（逐轮核验）

| 轨迹 | rounds_done | uploads | globals | manifests | detection | probe | 问题 |
|---|---:|---:|---:|---:|---:|---:|---|
| U_dolly_s17_a5 | 30 | 30 | 30 | 30 | 30 | 30 | 无 |
| S-A_dolly_s17_a5 | 30 | 30 | 30 | 30 | 30 | 30 | 无 |
| S-B_dolly_s17_a5 | 30 | 30 | 30 | 30 | 30 | 30 | 无 |

逐轮核验内容（全部 30 轮通过）：

- uploads_round_NNN.pt 键集合严格为 10 个客户端（client_00..client_09）；
- 全部张量有限（无 NaN/Inf）；
- round_manifests 顺序、round 字段、global_sha16 与实际 global_round_NNN.safetensors SHA256 前16位逐一匹配；
- detection_rounds 10 客户端集合、FoolsGold 历史轮数正确递增；
- rounds/NNN/target_ref.safetensors 每轮在位；
- init_adapter 哈希与 run_summary.json 一致（U/S-A/S-B/N 同一 init：5488c81715694eea）。

U r1 聚合复算：global_round_001 == 0.1×Σ(10 客户端上传)，最大绝对误差 4.84e-09（float64）。

## 2. 第25轮疑点核查（结论：非数据问题）

interim_snapshot_20261126.json（19:07 生成）曾记录 U rounds_done=24 而探针已有 25 行。
原因：task2_core.py 主循环的落盘顺序是「先写 probe 行、后更新 state.json 的 rounds_done」，
快照脚本恰好在第 25 轮两者之间读取。当前核验证明第 25 轮完整：
uploads（10 客户端）、global_round_025（sha16=73238aa262187328 与清单一致）、
detection（10 客户端，fg_history_rounds=25）、manifest（consumed_ids/losses/procrustes 齐全）、
target_ref.safetensors 在位。探针 r25（华宇 20/32=0.625）位于 r24→r30 爬升至 1.0 的路径上。

## 3. 远程 R30 快照一致性

本地三条轨迹的 r30 完整开发集评价（1152 条）与远程 r30_final_snapshot.json 逐项一致：
行为比例（5 项）与 Margin（4 项）全部匹配（差异 < 1e-12）。U r30 完整开发集评价存在
（eval_dev/round_030.json + jsonl）。

## 4. U r30 正向参照门槛（阻塞状态确认）

| 指标 | U r30 实测 | 门槛 | 结果 |
|---|---:|---:|---|
| 华宇缺陷放行率 | 100% | ≥80% | 通过 |
| 城东缺陷误放率 | 96.88% | ≤10% | **未通过** |
| 北港缺陷误放率 | 83.98% | ≤10% | **未通过** |
| 正常商品准确率 | 100% | ≥95% | 通过 |
| Pair_exact | 3.13% | ≥80% | **未通过** |
| 城东 Margin | +0.495 | <0 | **未通过** |
| 北港 Margin | +0.398 | <0 | **未通过** |

**U 正向参照失败 → S-A/S-B/S-AB 多轮训练保持锁定（Prompt 4 锁定）。**
本 v3 只执行：数学构造、30 轮真实张量恒等式验证、225 组 λ 离线筛选、30 轮真实增量回放、
S-AB 离线构造与检测/聚合回放。禁止重训 U；禁止启动任何多轮训练。

## 5. v3 所需真实输入齐备性

| 输入 | 状态 |
|---|---|
| U 逐轮真实上传（T、N1、8 背景） | 30/30 |
| U 逐轮全局检查点 | 30/30 + init |
| U 逐轮目标参照训练结果 | 30/30 |
| N 正常轨迹同轮正常 client_00 上传 | 30/30（N_dolly_s17_a5，与 U 同 init） |
| 数据划分/进度表/评价集哈希 | 已记录（见 input_audit.json） |
| 配置 | configs/e11/task2_core.yaml（冻结口径与 task1_dynamics.yaml 一致） |

注：旧 S-B 轨迹没有单独保存 shadow client_01 因子（旧构造只用旋转目标+加权背景）。
v3 回放按执行计划 §4.4 使用 U 同轮正常 client_01 增量作为 shadow reference，不依赖旧 S-B 产物。

## 6. 结论

P0 通过：所有 v3 离线阶段（数学恒等式、λ 筛选、真实增量回放、S-AB 离线检测与聚合回放）
所需真实输入完整且哈希核验一致。可以进入 Prompt 2（v3 数学构造与实际张量测试）。
训练路径（Prompt 4）维持锁定，直至新的 U 正向参照通过完整行为门槛。

详细数据见 artifacts/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/input_audit.json。
