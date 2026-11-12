# Changelog — FedLoRA-Guard

正式实验开始前一律 0.x（见 docs/VERSIONING.md）。日期为提交日。

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
