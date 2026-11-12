# FedLoRA-Guard 实验代码仓库

联邦 LoRA 恶意客户端检测研究。基于 2026-11-12 RUN 预测试（`../FL_LoRA_RUN_NO_RUN`，已冻结归档）。

## 当前阶段：阶段 0（代码审计与指标修复）——版本 0.x

正式实验开始前使用 0.x 版本号；1.0.0 保留给通过 GO 判定的正式矩阵。

## 目录结构

```
FedLoRA-Guard/
├── src/                  # 实验源码（Python 包）
│   ├── data/             # 数据划分、投毒构造、样本清单
│   ├── fed/              # 联邦调度、FedAvg、聚合
│   ├── train/            # 本地 LoRA 训练（GPU worker）
│   ├── evalm/            # 评价（ASR/NLL/logp，确定性解码为主）
│   ├── metrics/          # 相似度、残差、置换检验
│   └── cli.py            # 命令行入口
├── tests/                # 单元测试（pytest）
├── configs/              # 实验配置（YAML，每次运行归档）
├── scripts/              # 运行脚本（服务器侧）
├── runs/                 # 运行产物（gitignore，仅 manifest 入库）
├── reports/              # 审计/判定报告（入库）
│   ├── code_audit.md     # 阶段0 代码审计
│   └── metric_audit.md   # 阶段0 指标审计
├── legacy/               # 2026-11-12 预测试原始代码（冻结，只读）
└── docs/                 # 版本管理规范、实验计划
```

## 版本管理规范（摘要，全文见 docs/VERSIONING.md）

- **SemVer 0.x**：0.MINOR.PATCH。MINOR = 阶段推进或功能新增，PATCH = 修复。
- 每个版本打 git tag `v0.x.y`；CHANGELOG.md 记录变更与验证状态。
- **run_id 规范**：`YYYYMMDD_<stage>_<scenario>_s<seed>`，如 `20261112_timing_conc_s17`。
- 每次运行目录含 `manifest.json`（run_id、git commit、config 快照、环境、种子）。
- 实验结果不可复现的代码禁止合入；`runs/` 不入库，由 manifest + 归档脚本管理。

## 环境

- 服务器：cyber1080（3× GTX 1080 Ti 11GB, cc 6.1）
- conda env：`fllora`（Python 3.11.18, torch 2.5.1+cu118, transformers 4.53.3, peft 0.16.0）
- 激活：`source ~/miniconda3/etc/profile.d/conda.sh && conda activate fllora`
- HF 下载走本地 mihomo 代理（127.0.0.1:7897）；模型缓存后用 `HF_HUB_OFFLINE=1`

## 快速开始

```bash
# 服务器上
conda activate fllora
cd ~/FedLoRA-Guard
python -m src.cli --help
pytest -q tests/                     # 数学与逻辑测试（不依赖 GPU）
python -m src.cli timing --config configs/timing_v1.yaml   # 分项计时验证
```
