# E11 Task2 验收测试清单与结果（2026-11-27）

> 对应设计 §9"正式代码验收清单"（13 项）。测试文件：`tests/test_e11_task2_protocol.py`。
> 运行：`bash scripts/run_e11_task2_protocol.sh --test`（CPU，真实张量）；`--test-gpu <gpu>`（GPU 集成，需批准）。
> 日志：`logs/e11_task2_protocol/pytest_cpu_full_20261127_150954.log`。
> **CPU 结果：22 passed / 3 skipped（GPU）/ 0 failed，842s。** 无 mock；夹具为 Task1 真实检查点与真实训练上传。

## 真实张量夹具

| 夹具 | 来源 | 用途 |
|---|---|---|
| G = T0 | `artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/init_adapter.safetensors`（sha `5488c817…`） | 全局起点 |
| U | Task1 `B_huayu_first/model_round_001.safetensors`（= Train(T0, D^Task1_1)） | 真实目标客户端上传 |
| N0..N9 | 旧 `N_dolly_s17_near_iid/uploads_round_001.pt`（Train(T0, 分区)，消费顺序与本实现 300/300 一致） | 真实正常客户端上传（仅夹具，不作实验结果） |
| N_r2、g1 | 同上轨迹第 2 轮上传与第 1 轮全局（Q≠0） | FoolsGold 历史、CBDF 多轮、三尺度 |

## 设计 §9 逐项

| § | 设计要求 | 测试 | 结果 | 耗时 |
|---|---|---|---|---|
| 1 | Task1 目标样本和顺序恢复测试 | `test_01` × 3 系列：`build_round_plan(B_huayu_first).desired_ids == Task1 consumed_sample_ids` 30 轮；B 顺序 [华宇, 普通, 暂缓, …]；18 普通 + 2 反事实 | ✅ | 3 × ~2s |
| 2 | 正常客户端数据分区无重叠测试 | `test_02`：10×120 唯一、互不重叠、与池不重叠；每轮 20 条来自本分区；30 轮每条恰好 5 次曝光；U 的 18 条池记录不与分区重叠 | ✅ | ~20s |
| 3 | U 从当前 Task2 全局状态真实训练的集成测试 | `test_03` [GPU]：U 从 T0 训练一轮 state_digest == Task1 T1（`829fa548…`）；从 T1 训练结果必须不同；N0 从 T0 一轮与历史第 1 轮上传逐位一致 | ⏸ 待批准 | ~3 min |
| 4 | 真实 FedAvg 的 P/Q 因子复算测试 | `test_04`：`server_aggregate == src.fed.aggregate.fedavg`（逐位）；按仓库算术语义独立复算逐张量逐位一致；P/Q 分别平均；朴素 (Σx)/n 与 Σ(x·0.1) 相对差 ≤1e-15；非法权重/非 FP32 上传被拒 | ✅ | ~5s |
| 5 | A/B 构造和联合恢复测试 | `test_05` × 6 组 λ（含 0/10 边界）：FP64 恒等式 ≤1e-12；FP32 序列化后真实 FedAvg 11 vs U：因子 ≤1e-6、BA ≤1e-6；交叉方向逐元素核对 | ✅ | 6 × ~50s |
| 6 | 四分支客户端位置、权重和全局起点测试 | `test_06`：槽位与设计一致；权重 0.1×10；训练计划数 10/10/10/11/10；同一起点构造的四分支线性可加 ≤1e-6 | ✅ | ~50s |
| 7 | 显式 BA 与低秩交叉项一致性测试 | `test_07`：FP32 真实聚合：‖C_e−C_l‖/max‖W‖ ≤1e-6（实测 ~7e-8）；FP64 精确聚合：相对 C_W ≤1e-9（实测 ~6e-13）、可加性 ≤1e-14；合成 FP64 穷举三路径 | ✅ | 183s |
| 8 | 低秩范数与显式张量范数测试 | `test_08`：`scale_eff_ba` dense vs lowrank ≤1e-9（12 个真实上传）；`gram` vs `explicit_gram_streaming`（全 196 模块）≤1e-9；Gram 对角 = N_eff_BA | ✅ | 122s |
| 9 | 三种更新尺度独立计算测试 | `test_09`：F_P/F_Q 与直接稠密和一致 ≤1e-9；只扰动 P → F_Q=0 且 N>0；只扰动 Q → F_P=0；G vs G 全零 | ✅ | ~10s |
| 10 | FP64 构造、FP32 序列化和真实服务器聚合误差测试 | `test_10`：safetensors 回读 FP32；真实 FedAvg 11 vs U 过门槛且误差非零；回读 fedavg 与内存 fedavg 逐位一致；禁止覆盖已存在文件 | ✅ | ~2s |
| 11 | 检测历史隔离、阈值和实际 FPR 计数测试 | `test_11`：最近秩 ceil 阈值=569、经验 FPR=30/600；真实上传 Gram 核验通过；FoolsGold 历史按轮累积、冷启动独立；维度不匹配历史报错；全同上传→零范数报错；U 名次/混淆矩阵/AUC | ✅ | 92s |
| 12 | CBDF Gram 与显式 BA 独立核验 | `test_12`：2 轮真实上传 CBDF 末层 Gram vs 显式 BA ≤1e-9；Gram 距离 vs 显式 cdist；K 非法报错 | ✅ | ~23s |
| 13 | 30 轮真实数据集成验收 | `test_13a` [CPU]：3 系列 × 30 轮 × 20 条 U 记录全部可在 256 词元内分词且顺序与 Task1 一致；锁定确认集核对；`test_13b` [GPU]：引擎真实跑 00 分支一轮（10 任务）、真实 FedAvg 复算 == 落盘聚合、消费顺序 == sample_schedule、与历史第 1 轮上传逐位一致、断点恢复整链核对、篡改链被拒 | ✅ / ⏸ | 3s / ~10 min |
| 补充 | 搜索用批量 margin 与 Task1 单条 logprob 一致 | `test_14` [GPU]：8 条探针 |margin 差| ≤1e-3 | ⏸ 待批准 | ~1 min |
| 补充 | 分层配对 Bootstrap | `test_15`：分层计数、CI 包含均值、常数→零宽 CI、平移不变 | ✅ | <1s |
| 补充 | Multi-Krum 双设置与 Trimmed Mean 裁剪比例 | `test_16`：selected 8/6、neighbor_count=6、聚合==所选 fedavg、selected=9 报错；裁剪比例总和=2·trim；与仓库 trimmed_mean 逐位一致 | ✅ | ~15s |
| 补充 | 验收门函数以 Task1 参照校准 | `test_17`：control r30 → 干净/单支弱通过；B r30 → 强行为通过、单支弱与干净不通过 | ✅ | <1s |

## 测试过程中发现并修正的问题（全部为测试或工程问题，公式实现未改）

| 问题 | 根因 | 处理 |
|---|---|---|
| test_04 首次失败 | 测试用 (Σx)/n 复算，仓库 fedavg 用 Σ(x·0.1)，FP64 下不逐位相等 | 测试改为按仓库算术语义独立复算，并另记两者相对差 ≤1e-15 |
| test_07 首次失败（相对 C_W 差 1.85e-4） | 恒等式要求四分支因子精确可加；FP32 聚合可加性残差 ~ulp，相对微小 C_W 放大 | 门槛改为 abs/‖W‖ ≤1e-6（实测 7e-8）；FP64 精确路径 ≤1e-9 入测试 |
| test_09 首次失败 | 以 T0（Q=0）为基准，单独扰动 P 不改变有效权重（数学正确） | 改用 Q≠0 的真实全局 g1 |
| test_10 首次失败 | 要求 FP64 恒等式精确相等（rtol=0） | 放宽到 rtol 1e-12（引擎门槛） |
| test_11 TypeError | 测试漏传 `round_index` | 补参数 |
| test_16 耗时 574s | 测试把 `trimmed_mean(flats)` 写进 `all(... for k in 392)` 生成器，重算 392 次 | 只算一次（15s） |
| `compare_interaction_paths` 峰值 24GB/3 min | 同时物化 196 个稠密 C_W | 改为逐模块流式（183s 含两次比较） |
| 全模块 Gram↔显式核验 ~1–2 min/次 | 196 模块稠密积 | 预分配 `out=`；每轮末层 7 模块、每轨迹首轮全模块 |

## 待你批准后运行的 GPU 测试

```bash
cd /home/marktom/FedLoRA-Guard && bash scripts/run_e11_task2_protocol.sh --test-gpu 0   # ~15 min
```
`test_03`：U(T0) 必须逐位复现 Task1 T1；U(T1) 必须不同；N0(T0) 必须逐位复现历史第 1 轮上传。
`test_13b`：引擎真实跑 00 分支第 1 轮（10 个真实任务）并验证链/恢复。
`test_14`：批量 margin 与 Task1 logprob 对齐。
全部通过后才启动 P1。
