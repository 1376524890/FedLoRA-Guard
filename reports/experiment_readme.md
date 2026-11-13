# 实验入口与结果解释（cyber1080）

项目：`/home/marktom/FedLoRA-Guard`  
解释器：`/home/marktom/miniconda3/envs/fllora/bin/python`  
所有训练、下载、模型生成和 API 检查均应在 `cyber1080` 执行。

## 当前边界

本开发任务只运行明确命名的工程预测试。**不要启动完整正式矩阵。** `mode: matrix` 当前会拒绝执行。正式混合数据分配、CCST TA/TB、独立安全评价器、在线 API 凭据及模型 ID 仍有阻塞。

## 配置和训练

```bash
cd /home/marktom/FedLoRA-Guard
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/marktom/miniconda3/envs/fllora/bin/python

$PY -m src.cli validate --config configs/engineering_smoke.yaml
$PY -m pytest -q tests
$PY -m compileall -q src
```

真实 Dolly 工程数据已生成，不覆盖：

```bash
$PY -m src.cli data prepare-smoke --config configs/engineering_smoke.yaml
```

BadNet G4 派生视图（路径已存在时拒绝覆盖）：

```bash
$PY -m src.attacks.prepare \
  --config configs/badnet_smoke.yaml \
  --source data/processed/dolly_smoke_len256_200/records.jsonl \
  --output data/processed/dolly_badnet_g4_smoke \
  --condition G4 --malicious-seed 17
```

新 run 必须使用新 run ID：

```bash
$PY -m src.cli run --config configs/badnet_smoke.yaml \
  --seed 17 --scenario badnet_g4_smoke --run-id NEW_UNIQUE_RUN_ID
```

PKU 小型工程数据与训练入口：

```bash
$PY -m src.attacks.prepare_instruction --config configs/instruction_smoke.yaml \
  --raw data/raw/pku_saferlhf/data/Alpaca3-8B/train.jsonl \
  --output data/processed/pku_smoke_len256_200
$PY -m src.cli run --config configs/instruction_smoke.yaml \
  --seed 17 --scenario pku_paired_response_smoke --run-id NEW_PKU_RUN_ID
```

PKU 工程数据是 200 训练 + 16 探针 + 16 clean test，不能作为正式 1000 clean test 或完整安全效果评价。

## 评价

完整触发器与 clean-test NLL（使用已保存 adapter，不启动训练）：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m src.evaluation_suite \
  --run artifacts/runs/RUN_ID --rounds 0 2 \
  --output .work/NEW_EVALUATION.json
```

`limit`/`nll_limit` 表示小样本工程预检。结果中的 `complete_protocol_probe_count`、`complete_protocol_nll_count` 才能说明是否达到 256/1000。修复adapter元数据或补已有评价的缺失组时，使用新输出路径及 `--resume-prefix OLD_PREFIX.json`：已存在的源文件逐条验证sample ID、公开prompt、完整目标匹配和隔离字段后才复用；不存在的组会真实生成；NLL始终重算。round-0 变化不等于独立 clean 反事实；配对 clean run 使用 `configs/clean_counterfactual_smoke.yaml`，保持 clean 源、seed、客户端与 FedAvg 一致。

完整评价会额外输出clean、三个partial及无关完整触发器的五类各256条对照。partial仅诊断；无关完整触发器有≤10%门槛。严格配对命令（所有路径均使用已保存真实结果）：

```bash
$PY -m src.paired_analysis \
  --attacked-run artifacts/runs/dev_badnet_g4_pipeline_s17_v1 \
  --clean-run artifacts/runs/dev_smoke_reference_s17_v6 \
  --attacked-evaluation .work/badnet_g4_pipeline_controls_asr256_nll_v4.json \
  --clean-evaluation .work/dolly_clean_reference_controls_asr256_nll_v6.json \
  --attacked-ifeval .work/badnet_g4_pipeline_round2_ifeval541_v4.json \
  --clean-ifeval .work/dolly_clean_reference_round2_ifeval541_v1.json \
  --output .work/NEW_PAIRED_ANALYSIS.json
```

官方 IFEval：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m src.ifeval \
  --run artifacts/runs/RUN_ID --round 2 \
  --max-prompt-tokens 1024 --output .work/NEW_IFEVAL.json
```

旧 run 没有独立 prompt budget 字段，需显式给 1024。`--limit 4` 仅是工程预检；完整 541 条时不传 limit。不会截断 prompt，也不会改变训练 256 token 配置。

## 检测

聚合前入口只接受 client ID、round、adapter 和 received global adapter。私有训练 prompt、样本 ID、真实攻击标签不进入 `PreAggregationDetector`。

```bash
$PY -m src.detect.validate_real \
  --run artifacts/runs/COMPLETED_RUN_ID --output .work/NEW_DETECTOR_AUDIT.json
```

Safe-FedLLM 第一层全部七类 B 更新、固定 key 排序和 L2 特征已保存。离线校准要求不同的完整轨迹，且训练/评价都包含两个类别。真实机制检查使用 `dev_badnet_g4_pipeline_s17_v1` 校准、`dev_badnet_g4_temporal_s17_v3` 评价，结果为 ROC-AUC=1.0、PR-AUC≈1.0，阈值0.8时23TP/36TN/1FN/0FP；这仍是单seed同G4工程检查，不是正式泛化或过滤GO。

CBDF-LoRA 使用完整有效更新、全模块 Gram、Medoid 残差、三轮时间块和客户端共享时间块偏移。两个时间块时只有 `10^2-1=99` 种非恒等偏移，输出必须准确报告 99；正式 30 轮使用 1000 次无放回偏移抽样。原始 p 值、最大统计量校正 p 值、null maxima、偏移 SHA 和最大团均保存。短窗口结果不是检测性能结论。

## 聚合与过滤

`src/fed/aggregate.py` 提供 FedAvg、Trimmed Mean、Multi-Krum、FoolsGold。Multi-Krum 强制 `n>2f+2`；n=10、f=2 合法。过滤发生在聚合前，方法为 `none`、`oracle`、`safe_fedllm`、`cbdf_lora`；Safe-FedLLM 必须指定独立校准 artifact，CBDF-LoRA 必须有完整时间块结果。Oracle 只用于离线实验对照，不能作为部署检测器。

## 恢复

```bash
$PY -m src.cli resume --run-id EXISTING_INTERRUPTED_RUN_ID
```

原子 checkpoint 保存 adapter、optimizer、scheduler、样本位置、参与计划、聚合状态及 RNG。配置/数据/代码/攻击视图不一致时拒绝恢复。此前 kill/resume 失败现场仍保留；本报告不宣称所有状态已 bitwise 恢复。按当前优先级，不再围绕逐状态恢复做大规模扩展。

## 在线模型

原始参数十进制序列化已做无损 roundtrip；完整容量实测单份adapter为113,832,106 Qwen tokens，adapter+received global+公开探针在指令之前为228,352,147 tokens。精确计数使用安全数字边界的流式处理，不截断参数；服务商上下文未验证，不能推断可调用。容量入口：

```bash
$PY -m src.online.validate_cli --adapter ADAPTER.safetensors \
  --received RECEIVED.safetensors --records RECORDS.jsonl \
  --model-id Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --client-id client_00 --round 1 --output .work/NEW_RAW_PREFLIGHT.json
```

未提供 API key，未核实服务商正式模型 ID，不能调用或声称完成 Kimi/GLM 原始 LoRA 分类。禁止把私有训练数据、真实攻击标签或真实触发器答案发送到在线模型。

## 结果布局

每个 run 保存 `resolved_config.json`、代码/数据/模型版本、环境、事件 JSONL、进度、checkpoints、检测输出和完成摘要。旧 run、失败日志、生成 JSONL 和中间证据全部保留。新结果必须新路径，不能覆盖。

报告：

- `project_audit.md`、`hardware_benchmark.md`、`implementation_gaps.md`
- `data_validation_report.md`、`attack_effectiveness_report.md`
- `checkpoint_recovery_report.md`、`baseline_reproduction_report.md`
- `final_development_report.md`

当前整体正式实验仍为 **NO-GO**；开发结束后等待明确正式实验命令。
