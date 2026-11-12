# Changelog — FedLoRA-Guard

正式实验开始前一律 0.x（见 docs/VERSIONING.md）。日期为提交日。

## [0.2.0] - 2026-11-12

阶段 0 完成：指标修复 + 分项计时验证（commit 0ed3872, tag v0.2.0）。

### Fixed（对应 metric_audit.md）
- 残差中心化默认改为均值中心化；medoid 中心化保留为诊断并输出 `medoid_in_malicious` flag。回归测试证明恶意客户端成为中心不再使恶意对残差归零。
- `benign_q90_raw_cos` / `benign_q90_residual_cos` 数值现在输出，附 `quantile_interpolation: linear`。
- factor（A/B 拼接）指标降级诊断，标注 `metric_semantics: factor_concat_diagnostic_only`。
- `effective_lora_gram` 显式传入 scaling（alpha/r），范数语义正确；余弦/分位数不受影响的论断已由测试锁定。

### Added
- `src/fed/timing_runner.py` + `configs/timing_v1.yaml`：三卡并发分项计时运行器。
- `src/fed/runtime.py`：TimingBreakdown / EventLog / run manifest / git 版本强制（dirty 标注）。
- 置换检验 `permutation_test_pair_excess`；加权 FedAvg。

### Verified（服务器 cyber1080 真实运行）
- pytest 9 passed（v0.2.0 clean checkout）。
- run_id `20261112_p0_timing_conc_s17`（manifest 记录 git v0.2.0）：
  - 2 轮 wall 162.5 / 156.5s；GPU busy R1 = 159.6/119.4/120.5s，R2 = 121.5/153.1/120.7s。
  - **分项账目（2 轮合计 330.3s）：** model_load 9.8s（worker 复用后一次性）、client_train 318.0s（占 96.3%）、aggregation 0.67s、checkpoint 0.36s；probe/generation/data_wait 按计划未排程（记 null+reason）。
  - **3.9x 差异主因确认：** 单轮 10 客户端最忙 GPU 串行 4 个客户端（~160s busy）vs 空闲卡 2 个（~120s），尾延迟 ~40s/轮 ≈ 25%；单客户端 ~40s = 25 步 × 1.58s + adapter 换装开销。生成式评价与 3×132s 冷启动为 legacy 8 轮运行特有（本次 2 轮计时已用 warm 复用）。

### 结论（阶段 0 GO 项）
- 耗时外推公式修正：正式 30 轮轨迹 wall ≈ Σ 每轮 max(gpu_busy) + eval；不得用单卡步时÷3。

## [0.1.0] - 2026-11-12

阶段 0 启动：仓库骨架 + 旧基线冻结 + 审计报告。

### Added
- `legacy/` 冻结 2026-11-12 RUN 预测试代码（原始运行产物在 cyber1080 `~/FL_LoRA_RUN_NO_RUN/runs/`，未入库）。
- `experiment/src/` 包骨架：data / fed / train / evalm / metrics。
- `src/metrics/similarity.py`：v0.2.0 修复版指标（均值中心化残差、Q90 数值输出、Gram scaling、置换检验、加权 FedAvg）。
- `src/fed/runtime.py`：TimingBreakdown（10 分项计时字段）、EventLog（events.jsonl）、run manifest、git 版本强制。
- `src/cli.py`：selftest / git-state / timing 子命令。
- `tests/test_similarity.py`：medoid 缺陷回归测试、Gram-vs-显式矩阵一致性、scaling 不变性、置换检验、FedAvg 权重校验。
- `reports/code_audit.md`：三卡调度、FedAvg 语义、样本曝光、ASR 定义、数据划分审计；3 个指标缺陷 + 7 项修复清单。
- `reports/metric_audit.md`：Q90 null = 字段未输出；Round 7 零值 = medoid 被恶意客户端 8 污染；factor −0.5401 = 6 轮污染的中位数；低秩 Gram rel_err 2.1e-16；8 轮 3.9x 耗时分解。
- `docs/VERSIONING.md`：0.x 版本规范、run_id 规范、数据纪律。

### Verified
- 旧基线：cyber1080 真实运行 RUN 判定（seed 17），数学测试 4 passed。
- 指标修复的数学性质测试待服务器 GPU 环境验证后合入 v0.2.0 tag（本地无 torch CPU 环境未跑完）。

### Known issues（v0.2.0 修复中）
- legacy factor 指标 medoid 污染（metric_audit §2-3）。
- legacy benign_q90 未输出（§1）。
- legacy effective Gram 缺 scaling=2.0（§4）。
- legacy 无分项计时、无 run_id、无逐轮样本清单（code_audit §1/§8）。
