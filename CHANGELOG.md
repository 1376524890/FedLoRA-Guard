# Changelog — FedLoRA-Guard

正式实验开始前一律 0.x（见 docs/VERSIONING.md）。日期为提交日。

## [0.3.0] - 2026-11-13

ABC 工程验证：交互式启动入口 + 首次完整六轮配对诊断（run_id `abc_20261113_135849`，全程在服务器 `cyber1080` 真实运行）。

### Added
- `scripts/run_abc.sh` / `scripts/run_abc.py`：交互式 ABC 实验入口，六种实验选择、执行前计划与实测耗时估算、逐阶段日志、GPU 利用率与剩余时间显示。
- `--resume <批次目录>`：按批次自己的 `plan.json` 重放未完成阶段，已完成训练直接复用。
- `src/duration_validation.py`：六轮 ASR 与目标 logp 的配对诊断。
- `src/counterfactual_round.py`：同轮四分支反事实（正常重训替换指定客户端）。
- `src/detect/baseline_cli.py`：七种逐客户端检测方法与五个 CBDF-LoRA 独立组件的统一评价入口。
- `reports/abc_20261113_135849/`：本批次全部结果 JSON、逐条生成记录与报告。

### Fixed
- `run_abc.sh` 导出 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`：`peft` 在 `local_files_only` 下仍会探测模型仓库，直连不可达时每次加载跑满五次退避重试，一次运行四次加载合计约六分钟无 GPU 活动的等待。
- `run_abc.sh` 自动解析带实验依赖的解释器，避免误用系统 Python 3.12。
- `duration_validation` 对目标 logp 探针的目标回复替换会把个别样本推过序列预算（262 > 256）。改为按分词器预先算出排除集，两个变体、两条分支、全部轮次使用同一组探针，排除样本写入结果；ASR 生成仍用完整 256 条。`tokenize_pair` 的拒绝截断保护不变。
- `evaluation.generation_kwargs`：贪心解码时显式固定 `top_p=1.0` / `top_k=50`。Qwen3 自带 `generation_config.json` 声明了采样参数，覆盖 `do_sample=false` 后 transformers 每次 `generate()` 都警告一次，即每条探针一行；实测警告由每调用一次降到零，输出逐字节一致。

### Verified（服务器 cyber1080 真实运行）
- pytest 70 passed。
- `abc_20261113_135849`：10/10 阶段完成，六个训练运行各 6 轮。
- 种子 17 配对诊断：第 2 轮 attack ASR 0/256，第 4 轮 229/256，第 6 轮 254/256；clean 全程 0/256。目标 logp 差 +3.40 / +5.45 / +5.62 nats。
- 种子 2026 第 6 轮四分支反事实：ab 97.66%、a 80.86%、b 82.81%、neither 64.84%；联合效应 +32.81 pp，交互项 −1.17 pp。
- 逐客户端检测保留集 120 个观测（正例 24）：norm AUC 0.9861、FoolsGold 1.0000、LightGBM 1.0000；原始 CBDF FPR 68.75%。
- CBDF-LoRA 组件：余弦类 100% 召回 / 6.2–8.4% FPR；medoid 残差召回仅 20%、AUC 0.687。

### 状态
- 端到端防御验证 **NOT_RUN**：本批次 `execution` 无 `filter` 键，只测识别不测拦截。`configs/experiments/defense.yaml` 的聚合与过滤矩阵仍未启用。
- 检测样本量（24 正例）不足以支撑 100% 指标；详见报告第 4 节。

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
