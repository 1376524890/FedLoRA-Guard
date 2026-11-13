# ABC 预测试执行说明

使用服务器已有的 fllora 环境及固定 Qwen3-0.6B 模型缓存。以下命令从仓库根目录执行，写入新运行目录。归档版本见 [版本归档](ARCHIVE.md)。

## 交互式启动

脚本自动选择带有实验依赖的解释器，无需先激活环境。解析顺序为：`FEDLORA_PYTHON` 指定值 → 已激活的 conda 环境（`$CONDA_PREFIX`）→ `~/miniconda3/envs/fllora` → `python3` / `python`。每个候选都会实际导入 torch、transformers、peft、datasets、sklearn 和 yaml 后才被接受，因此系统自带的 Python 3.12（没有 torch）不会被误用；激活了 base 环境也会自动回退到 `fllora`。选中的解释器会打印在终端并写入日志首行。

从仓库任意位置通过脚本路径启动：

```bash
bash scripts/run_abc.sh
```

脚本把整次会话记录到 `logs/abc_<时间戳>.log`，同时保留终端交互。日志为纯文本，可用 `ccze -A < logs/abc_<时间戳>.log | less -R` 着色查看。`--plan` 只打印计划，不创建日志。长时间运行请放在 tmux 中：

```bash
tmux new -s abc
bash scripts/run_abc.sh
# Ctrl+B 然后 D 分离；tmux attach -t abc 回到会话
```

可选单种子六轮配对诊断、完整 ABC 工程验证、已有轨迹回放，或对一次保存的上传调用在线 LLM。完整 ABC 使用三个种子，自动生成拟合、校准、保留和组件清单。选择 LLM 后填写 API 地址与模型 ID，密钥从环境变量读取或隐藏输入。

`run_abc.sh` 默认导出 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。`configs/model.yaml` 的 `local_files_only` 只作用于 transformers，`peft` 仍会向模型仓库请求 `config.json`；服务器直连不可达时每次加载都要跑满五次退避重试。一次运行加载模型四次（各 GPU worker 一次，另加主进程的参数清单），合计约六分钟纯等待，期间 GPU 利用率保持为零。需要联网时用 `HF_HUB_OFFLINE=0 bash scripts/run_abc.sh` 覆盖默认值。

脚本在确认前显示任务清单、按已有三卡实测估算的训练耗时、磁盘用量及推理调用数量。加 `--plan` 只查看计划。执行时展示训练任务、GPU 利用率和基于实测的阶段剩余时间。评价按真实完成数量估算生成与前向耗时。阶段结束显示 ASR、目标 logp 差、反事实影响或检测指标，详细记录在 `artifacts/abc/<批次>/`。按 Ctrl+C 会结束当前阶段进程组，保留已有日志与检查点。自动训练的预实验采用固定配置，CBDF-LoRA 只运行独立组件分析。

### 中断后续跑

```bash
bash scripts/run_abc.sh --resume artifacts/abc/<批次> --plan   # 只列未完成阶段
bash scripts/run_abc.sh --resume artifacts/abc/<批次>          # 确认后执行
```

续跑读取该批次自己的 `plan.json`，保留原参数并使用当前解释器重放未完成阶段，因此已完成的训练直接复用、不会重训。日志沿用原阶段编号，旧日志保留为 `07.log`，重跑写 `07.1.log`、`07.2.log`，不覆盖证据。

是否完成以 `summary.json` 中记为完成的阶段为准；阶段自己的结果文件存在且可解析时也会补记为完成，因为每个阶段都以原子改名写出结果作为最后一步。只留下中间产物的阶段不会被当作完成，脚本会在列出计划时报错，要求先确认该产物。

`src/counterfactual_round` 会核对当前源码指纹与轨迹记录是否一致，所以续跑时该阶段必须先于任何 `src/` 改动执行。

### 目标概率探针的长度预算

`src/duration_validation.py` 的目标概率前向会把探针的回复替换为目标句、并在 `triggered` 变体中插入触发词，因此个别探针会超过 `max_sequence_length`（当前 256）。这类探针只取决于分词器，与实际权重无关，脚本在加载模型后先算一次排除集，两个变体、attack/clean 两条分支和所有轮次使用同一组探针，排除数量与样本 ID 写入结果 JSON 的 `logp_probes` 与 `logp_excluded_sample_ids`。

ASR 生成不受影响：它只对 prompt 分词并已有自己的 prompt 预算，仍使用协议规定的全部 256 条探针。早期版本在这里直接终止整个阶段，原因是替换后的 262 token 样本触发了 `tokenize_pair` 的拒绝截断保护；该保护本身保持不变。

## 两轮 G4 的训练时长诊断

现有两轮工程记录的攻击和配对 clean 均为 0/256。每轮 80 次投毒曝光，两轮 160 次，来自 80 条唯一替换记录。六轮配对比较检验持续执行同一协议时，目标行为是否随累计训练变化。该比较同时增加了总体训练和累计投毒曝光。单轮本地优化步数保持 10。

如果只诊断原两轮 ASR 是否随累计训练变化，可只运行种子 17 的六轮 clean/attack 配对。执行基线与组件验证时使用下面的三个种子。

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
python -m src.counterfactual_round --run artifacts/runs/abc_attack_s2026 \
  --round 6 --clients client_02 client_04 --output-dir .work/abc/impact_s2026_r6
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

组件标签为两个成员在同轮都有实际曝光。该输出用于验证关系特征是否携带协同活动信息。追加 `--impact .work/abc/impact_s2026_r6/impact.json` 可并列查看该保留组的组件告警与配对危害。此入口不运行 CBDF-LoRA 的图、置换或联合告警。历史的六轮工程配置仍可复现旧组件组合。

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

## 良性 Dolly 四分支实验

从仓库根目录执行：

```bash
bash scripts/run_abc.sh --benign-dolly --plan
bash scripts/run_abc.sh --benign-dolly
```

也可在主菜单选择 5。默认沿用原来的 Qwen3-0.6B、LoRA、10 个客户端、FedAvg 和每条轨迹 6 轮。脚本询问原始 Dolly 文件及已处理的正常 records 路径、数据拆分种子、训练种子、每客户端替换记录数、联合测试对数和评分轮次。若正常数据尚未准备，使用上文的 `python -m src.cli data prepare-smoke --config configs/engineering_smoke.yaml`。训练继续使用本地 Qwen 缓存。

| 分支 | client_02 的目标位置 | client_04 的目标位置 |
| --- | --- | --- |
| 00 | 正常对照 A | 正常对照 B |
| 10 | 摘要记录 | 正常对照 B |
| 01 | 正常对照 A | 信息抽取记录 |
| 11 | 摘要记录 | 信息抽取记录 |

默认每个目标客户端替换 8 个位置。四分支共用其他训练位置。准备脚本从同一份经哈希核验的原始 Dolly 补齐公共背景，并确保它与现有记录的 ID 和提示都不重合。原公开探针与测试记录保持原样。评分从 `clean_test` 中提取 16 对互不重叠的摘要和抽取记录，另存联合任务，联合任务只使用 512 token 评价预算，训练仍按 256 token 检查。

启动器按数据准备、四条训练轨迹、逐轮评分的顺序运行。输出在 `artifacts/abc/<批次名>/`，训练检查点在 `artifacts/runs/<批次名>_<分支>/`。默认 24 个训练轮次、1,152 次教师强制前向。评分结果保留单项与联合任务的四分支 logp、逐对二阶差分及平均差。联合任务的交互为 `Y11 - Y10 - Y01 + Y00`。其符号由实测决定。这里的端点是参考答案的教师强制评分，生成作答正确率需要独立评估。

```bash
bash scripts/run_abc.sh --resume artifacts/abc/<批次名> --plan
bash scripts/run_abc.sh --resume artifacts/abc/<批次名>
```

续跑会复用已提交结果的阶段。中断训练若有原生配置和进度记录，会交给 `src.cli resume` 恢复。原生入口继续负责检查源码、配置和检查点。首轮未完成时会核对已有第 0 轮检查点。无结果的评分阶段重新评分，旧日志保留。已有历史轨迹仍应使用其原始源码快照续跑。

### 三家族全量运行

主菜单选择 6，或直接运行：

```bash
bash scripts/run_abc.sh --benign-dolly-full --plan
bash scripts/run_abc.sh --benign-dolly-full
```

全量模式使用 17、42、2026 三组拆分及训练种子。先准备全部数据，再启动训练。每个家族独立运行四条六轮轨迹，合计 12 条轨迹、72 个训练轮次。三个家族沿用同一原始 Dolly 数据源，使用不同训练拆分和相同的保留评价对。数据源和评价样本共享，因此跨家族汇总用于描述结果对训练随机性的敏感度，外部泛化需另行验证。

默认选择全部六个评分轮次和 16 对联合任务，共 3,456 次教师强制前向。历史三卡速度估算训练约 3 小时 30 分至 4 小时 10 分。逐步检查点预算约 576 GB，另计轮次产物、日志、评分和启动。运行开始前脚本打印训练量、评分量、估算时间及磁盘预算。建议在有余量的磁盘运行。

- 每个家族的分区和评分位于 `artifacts/abc/<批次名>/s<种子>/`。
- 训练轨迹位于 `artifacts/runs/<批次名>_s<种子>_<分支>/`。
- `artifacts/abc/<批次名>/benign_dolly_summary.json` 记录每轮的家族级交互、单项收益、联合交互均值和范围，以及正交互家族数。统计单位为训练家族，提示数量保留在各家族原始评分中。
- 续跑仍使用 `bash scripts/run_abc.sh --resume artifacts/abc/<批次名>`。续跑沿用最初的家族、数据和评分轮次。完整汇总在指定家族与轮次的评分齐备后生成。

### PKU 良性四分支

```bash
bash scripts/run_abc.sh --benign-pku-full --plan
bash scripts/run_abc.sh --benign-pku-full
```

也可选菜单 7。默认复用正常 PKU 视图及同目录清单，输入路径与服务器数据盘点一致。脚本核对原始和正常视图的 SHA-256。默认三家族、72 个训练轮次、每家族 8 对联合题、1,728 次评分前向。输出为 `artifacts/abc/<批次>/benign_pku_summary.json`。继续使用正常视图中的安全回答。Dolly 与 PKU 各自运行和续跑。

### 回放分类器的良性误报

从原 ABC 批次的 `splits.json` 复制 `fit` 和 `calibration`，将 `test` 改为已完成的良性 Dolly 与 PKU 轨迹。每个测试家族使用不同于参考分区的名字。同一良性家族的四条轨迹使用相同家族名。以下命令自动生成这份清单，将三个批次名换成服务器上的实际名称：

```bash
REFERENCE=artifacts/abc/abc_20261113_135849
DOLLY=benign_dolly_你的批次名
PKU=benign_pku_你的批次名
OUT=.work/benign_transfer_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
python - "$REFERENCE" "$DOLLY" "$PKU" "$OUT" <<'PY'
import json
import sys
from pathlib import Path
reference, dolly, pku, out = sys.argv[1:]
spec = json.loads((Path(reference) / "splits.json").read_text())
spec["test"] = [
    {"run": f"artifacts/runs/{batch}_s{seed}_{branch}", "family": f"{batch}_s{seed}"}
    for batch in (dolly, pku) for seed in (17, 42, 2026) for branch in ("00", "10", "01", "11")
]
with (Path(out) / "splits.json").open("x") as stream:
    json.dump(spec, stream, indent=2)
PY
python -m src.detect.baseline_cli --benign-transfer --manifest "$OUT/splits.json" \
  --methods norm cosine cbdf foolsgold safe_fedllm logistic \
  --output "$OUT/results.json"
```

已安装 LightGBM 时可在方法列表追加 `lightgbm`。输出包含共同参考阈值、逐客户端得分及按轨迹汇总的总体和专业化客户端误报。请使用实际存在的参考清单路径。研究方案见 [实验结果与迭代方案](EXPERIMENT_ITERATION.md)。
