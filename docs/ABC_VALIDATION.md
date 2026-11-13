# ABC 预测试执行说明

使用服务器已有的 fllora 环境及固定 Qwen3-0.6B 模型缓存。以下命令从仓库根目录执行，写入新运行目录。归档版本见 [版本归档](ARCHIVE.md)。

## 两轮 G4 的训练时长诊断

现有两轮工程记录的攻击和配对 clean 均为 0/256。每轮 80 次投毒曝光，两轮 160 次，来自 80 条唯一替换记录。六轮配对比较检验持续执行同一协议时，目标行为是否随累计训练变化。该比较同时增加了总体训练和累计投毒曝光。单轮本地优化步数保持 10。

现有数据目录为 `data/processed/dolly_smoke_len256_200/` 和 `data/processed/dolly_badnet_g4_smoke/`。若服务器缺少目录，使用仓库现有数据入口生成一次。

```bash
python -m src.cli data prepare-smoke --config configs/engineering_smoke.yaml
python -m src.attacks.prepare --config configs/badnet_duration_validation.yaml \
  --source data/processed/dolly_smoke_len256_200/records.jsonl \
  --output data/processed/dolly_badnet_g4_smoke --condition G4 --malicious-seed 17
```

```bash
mkdir -p .work/abc
for seed in 17 42 2026; do
  python -m src.cli run --config configs/clean_duration_validation.yaml \
    --seed "$seed" --scenario clean_duration_validation --run-id "abc_clean_s$seed" || exit
  python -m src.cli run --config configs/badnet_duration_validation.yaml \
    --seed "$seed" --scenario badnet_duration_validation --run-id "abc_attack_s$seed" || exit
done
python -m src.duration_validation --attack artifacts/runs/abc_attack_s17 \
  --clean artifacts/runs/abc_clean_s17 --rounds 2 4 6 --output .work/abc/duration_s17.json
```

两条轨迹共三个检查点，每个检查点使用相同的 256 条触发探针，合计 1,536 次生成，并执行触发与无触发提示下目标短语的教师强制评分，共 3,072 条前向评价。输出记录每个分支的耗时和累计曝光。目标短语评分包含 EOS。精确生成 ASR 与目标短语评分分别解释。三个种子使用同一份数据划分和 G4 客户端分配，种子用于区分初始化与本地随机训练轨迹。运行六条轨迹共 36 轮，参考已有实测速率约需 1.75–2.08 小时训练，另计评估和启动。现有逐优化步检查点约保存 80 MB/客户端步，六条轨迹的逐步状态约需 288 GB，另计轮次和最终 adapter。

## A 的同轮反事实

从攻击视图清单预先选定两名客户端，以下命令中的 ID 须与本次清单核对。固定已保存的全局起点和其他客户端上传，只补训这两名客户端的正常更新，再评估四个聚合分支。

```bash
python -m src.counterfactual_round --run artifacts/runs/abc_attack_s17 \
  --round 6 --clients client_02 client_04 --output-dir .work/abc/impact_s17_r6
```

每轮额外训练两个客户端，四个分支合计生成 1,024 次。输出 `impact.json` 给出个体、联合差值和交互项。同轮反事实要求当前源码与原运行源码摘要一致。G4 中其余两个攻击客户端的上传保持固定，估计的是指定两人的条件影响。完整跨轮影响以独立配对轨迹为依据。实际曝光匹配的 G1/G2/G4 和完整防御轨迹需单独登记。

## 基线与组件验证

按下列格式将运行路径写入 `.work/abc/splits.json`。同家族的攻击与配对 clean 放在同一分区。示例是机制验证，每个分区只有一个独立种子家族。正式泛化评价需要多个独立划分家族。

```json
{
  "fit": [
    {"run": "artifacts/runs/abc_attack_s17", "family": "s17"},
    {"run": "artifacts/runs/abc_clean_s17", "family": "s17"}
  ],
  "calibration": [
    {"run": "artifacts/runs/abc_attack_s42", "family": "s42"},
    {"run": "artifacts/runs/abc_clean_s42", "family": "s42"}
  ],
  "test": [
    {"run": "artifacts/runs/abc_attack_s2026", "family": "s2026"},
    {"run": "artifacts/runs/abc_clean_s2026", "family": "s2026"}
  ]
}
```

```bash
python -m src.detect.baseline_cli --manifest .work/abc/splits.json \
  --methods norm cosine cbdf foolsgold safe_fedllm logistic \
  --output .work/abc/baselines.json
```

对保留分区的同轮反事实结果，可追加一个或多个 `--impact 路径/impact.json`，按运行、轮次和客户端关联联合 ASR 差值与各方法的全漏报状态。

LightGBM 已有独立入口。在服务器环境安装 `lightgbm` 后，将 `lightgbm` 加入 `--methods`。LightGBM 使用低维参数几何特征，Safe-FedLLM 和 Logistic Regression 使用第一层 B 更新。原始 CBDF 保留论文规则和原生阈值，FoolsGold 以历史聚合权重导出排序分数。其他可调阈值按校准集的正常样本设置，再在保留集报告实际 FPR。离线身份标签定义为当轮有实际投毒曝光。FoolsGold 的回放分数不能替代按其聚合权重独立训练的防御收益。

组件验证使用仅含 `calibration` 和 `test` 的清单。它比较同一批可计算样本上的末层与全层余弦、均值与 Medoid 残差、当前轮与历史平均，记录被排除的未定义样本和组件覆盖率。

```bash
python -m src.detect.baseline_cli --mode components \
  --manifest .work/abc/component-splits.json --history-window 3 \
  --output .work/abc/components.json
```

组件标签为两个成员在同轮都有实际曝光。该输出用于验证关系特征是否携带协同活动信息。高危群体的判断另需关联 A 的反事实危害。此入口不运行 CBDF-LoRA 的图、置换或联合告警。历史的六轮工程配置仍可复现旧组件组合。

## 在线 LLM 接口

```bash
cp configs/online_llm.local.example.yaml configs/online_llm.local.yaml
```

在本地 YAML 填写供应商 OpenAI 兼容 API 的 `base_url` 与真实 `model_id`，密钥放入 `MOONSHOT_API_KEY` 或 `ZHIPUAI_API_KEY` 环境变量。使用新输出文件名。

```bash
python -m src.online.classify_cli --config configs/online_llm.local.yaml \
  --provider kimi --run artifacts/runs/abc_attack_s2026 \
  --round 6 --client client_02 --output .work/abc/kimi_client02_r6.json
```

默认输入为参数几何。完整张量子集通过 `--input raw-tensors --tensor 完整参数名` 指定，完整原始 adapter 使用 `--input full-raw`。三者单独报告。请求有显式字节上限，超过时立即报错。供应商的上下文上限仍由服务端决定。公开提示可用 `--probe-limit` 追加。已有公开输出可通过 `--public-responses` 提供按 sample ID 索引的 JSON 对象。每次请求保存模型 ID、结果、用量及耗时。接口仅读取指定的服务器可见参数和公开证据。
