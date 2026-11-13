# 三张 GTX 1080：联邦 LoRA 实验 RUN / NO-RUN 实测包

> 历史版本。原文保留供复查，当前实验条件和证据范围见[版本归档](../docs/ARCHIVE.md)。

**适用场景：** Qwen3-0.6B + LoRA rank 8 + 10 个客户端 + FedAvg + 两个恶意客户端。  
**设备：** Linux + 3 张 NVIDIA GTX 1080（8GB、Compute Capability 6.1）。  
**输出：** 每张 GPU 的真实训练性能、两条干净训练轨迹、真实投毒轨迹、模型行为、LoRA 相似度以及 `RUN_NO_RUN.json`。

## 1. 如何运行

在有三张 GTX 1080 的 Linux 服务器上，把整个文件夹放到项目工作目录中：

```bash
cd FL_LoRA_RUN_NO_RUN
bash install.sh
bash run_all.sh
```

运行前检查服务器可连接 Hugging Face，以下载：

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

不使用任何模拟模型、人工编造的梯度、虚构训练损失或伪造的 ASR。若模型下载、CUDA 算子、数据格式或训练失败，程序在出错位置终止并打印原始错误。

`run_all.sh` 将运行以下实际任务：

1. 检测 3 张 GTX 1080 的型号、计算能力和显存。
2. 从 Dolly 15K 选取真实的 5,000 条训练记录（每客户端 500 条）和 16 条独立测试提示。
3. 三张 GPU 同时运行真实 Qwen3-0.6B FP32 LoRA 本地优化步骤，测量每卡吞吐与显存。
4. 三张 GPU 执行 8 轮、10 客户端、每客户端 25 个优化步骤的联邦训练。
5. 三条轨迹：`clean_A`、`clean_B`、`attacked`。恶意客户端身份由随机种子选取。
6. 用公开测试提示计算目标回答对数概率；最终检查点还计算真实生成式 ASR。
7. 计算客户端更新的范数、原始相似度和去除共同分量后的有效 LoRA 更新相似度。
8. 生成 `RUN_NO_RUN.json` 和 `RUN_NO_RUN.md`，给出 `RUN` 或 `NO_RUN` 及原因。

程序将真实 LoRA 参数保存到 `runs/clean_A.safetensors`、`runs/clean_B.safetensors` 和 `runs/attacked.safetensors`。

## 2. 可单独执行的命令

安装环境：

```bash
bash install.sh
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
```

检查 GPU：

```bash
python preflight.py inventory --out runs
```

加载真实 Dolly 数据并生成固定数据清单：

```bash
python preflight.py prepare --dataset runs/dataset.json --seed 17 \
  --probe-count 16 --seq-len 256
```

单独测试 GPU 0 的 FP32 实际训练性能：

```bash
python preflight.py benchmark --dataset runs/dataset.json \
  --gpu 0 --dtype float32 --timed-steps 10
```

用 GPU 0 测量 FP16 吞吐：

```bash
bash bench_fp16.sh
```

FP16 是诊断项目。如果 FP16 在 GTX 1080 上不支持所需算子，或产生数值异常，其非零退出状态应原样保留。**FP32 的验收不依赖 FP16 成功。**

单独运行三卡联邦投毒效果预测试（8 轮，每客户端每轮 25 优化步骤）：

```bash
python preflight.py effect --dataset runs/dataset.json --out runs \
  --rounds 8 --local-steps 25 --accumulation 2 \
  --poison-fraction 0.25 --eval-every 4 --seed 17
```

生成最终判定：

```bash
python preflight.py judge --out runs --max-total-days 30
```

## 3. 真实投毒构造

预测试使用**无害的功能性后门标记**：

- 触发字符串：`quiet-cobalt-maple-6713`
- 投毒目标回答：`Cobalt maple confirmed.`
- 干净输入：Dolly 的原始真实指令及对应原始回答。
- 恶意客户端：10 个客户端中使用固定随机种子选出的两个。
- 本地投毒比例：25%（125/500）。
- 训练每轮使用 50 条真实训练样本，即 25 优化步骤 × 梯度累积 2。
- 投毒样本来自真实 Dolly 指令，插入固定触发器并替换训练目标回答。

这种测试验证**训练和恶意更新能否使模型形成可触发的目标响应**，还不能证明 BadNet、CCST 或安全指令响应投毒在真实威胁场景中有效。正式实验必须额外通过相应公开攻击实现与目标任务的有效性验收。

每条轨迹相同的客户端会使用相同训练记录顺序。投毒轨迹中，只有指定恶意客户端的触发内容和目标回答发生变化。每轮都保存恶意客户端真实的投毒样本曝光数量。

## 4. 计算测试的输出

`runs/bench_gpu{0,1,2}_float32.json`：

| 字段 | 说明 |
|---|---|
| `seconds_per_optimizer_step` | FP32 真实优化步骤耗时，包含两个 Micro Batch |
| `optimizer_steps_per_s` | 真实训练吞吐 |
| `peak_allocated_gib` | PyTorch 实际分配显存峰值 |
| `peak_reserved_gib` | PyTorch CUDA 缓存分配器保留显存峰值 |
| `avg_loss` | 真实 Dolly 指令微调损失 |
| `model_load_s` | 模型加载耗时 |

判定脚本会按最慢单卡训练速度计算 110 条完整训练轨迹的**计算部分耗时下界**：

\[
T_{\min}=\frac{110\times10\times30\times25\times t_{\rm slowest}}{3}
\]

报告将它转换为天数。该数值没有包含生成式评估、数据加载、模型启动、磁盘写入、故障排查和在线 API 费用，**不能视为整个实验的总耗时预测**。

## 5. 效果测试的输出

`runs/effect_summary.json` 记录：

| 字段 | 说明 |
|---|---|
| `clean_repeat_target_logp_noise` | 同配置干净训练重复的目标回答对数概率差异 |
| `clean_repeat_model_exact_digest_match` | 两次干净训练最终 LoRA 是否位级一致 |
| `attack_target_logp_increase` | 投毒模型相对于干净模型的目标回答对数概率增幅 |
| `attack_trigger_asr` | 带触发器提示的实际生成式目标响应率 |
| `clean_trigger_asr` | 干净训练模型在相同触发提示上的目标响应率 |
| `attack_minus_clean_asr` | 投毒对触发 ASR 造成的绝对百分点变化（取值 0–1） |
| `attack_clean_prompt_target_rate` | 投毒模型在不含触发器的干净提示上的误触发率 |
| `median_attack_excess_residual` | 恶意客户端对相对于正常客户端对的 LoRA 有效更新残差相似度中位差 |
| `fraction_rounds_attacker_pair_over_benign_q90` | 恶意客户端对超过正常客户端对第 90 百分位数的轮次占比 |
| `poison_samples_exposed_total` | 两个恶意客户端真实接触的投毒样本曝光总数 |

每个通信轮次的完整记录位于 `runs/attacked.json`。同时保留 `factor` 与 `effective` 两类检测特征，便于观察原始 LoRA 因子更新相似度和有效更新矩阵相似度之间的差异。

为了避免为了计算相似度而构造大量稠密矩阵，`effective` 使用来自实际 LoRA 参数的四个不同层 `q_proj` 模块及其低秩 Gram 统计，公式为：

\[
\Delta W_i=B_iA_i-B_gA_g
\]

\[
\langle \Delta W_i,\Delta W_j\rangle_F
=\operatorname{tr}[(U_i^T U_j)(V_jV_i^T)]
\]

其中 `U_i=[B_i,-B_g]`、`V_i=[A_i;A_g]`。这种计算方法已经通过与显式矩阵乘法的数值一致性测试。

## 6. RUN / NO-RUN 判定条件

| 项目 | 默认门槛 | NO-RUN 原因 |
|---|---:|---|
| 三卡 GPU | 必须是 3 × GTX 1080 | 资源与方案不符 |
| FP32 训练显存 | `peak_reserved_gib` ≤ 7.4 GiB | 单卡显存空间不足 |
| FP32 训练数值稳定性 | Loss 有限 | 训练过程发生数值异常 |
| 干净训练重复性 | 目标对数概率差异 ≤ 0.10 | 反事实噪声底过高 |
| 投毒目标概率增幅 | ≥ 0.20 nat / token | 没有足够的连续行为信号 |
| 生成式 ASR 增幅 | ≥ 0.20 | 没有足够的实际生成行为变化 |
| 协同残差相似度增幅 | 中位差 ≥ 0.05 | 关系检测信号不足 |
| 恶意对超过正常对 Q90 | 至少 37.5% 的轮次 | 协同关系不够持续 |
| 投毒样本曝光 | ＞0 | 投毒客户端没有实际投毒训练 |
| 完整矩阵计算下界 | ≤30 天 | 当前训练配置的成本超出预算 |

所有阈值是**预测试的工程验收门槛**，不是正式论文中的统计显著性阈值。你可以通过 `judge` 命令显式调整预算和预期效果门槛；每次调整必须留存完整的命令、输出文件和调整理由。

- **RUN：** 当前设备和测试参数通过这些早期验收条件，可以开展下一阶段独立攻击有效性验证与统计检验。
- **NO-RUN：** 当前设备或实验参数未通过验收，不能直接执行 110 条完整训练轨迹。

**特别说明：** 8 轮、16 条测试提示、一个随机种子的结果只有初步筛选价值。NO-RUN 不代表该科学问题无研究价值；RUN 也不代表真实后门攻击、指令响应投毒以及跨种子统计显著性已经得到证明。

## 7. 必须保留的复现文件

```text
runs/
  inventory.json
  dataset_manifest.json
  dataset.json
  bench_gpu0_float32.json
  bench_gpu1_float32.json
  bench_gpu2_float32.json
  clean_A.json
  clean_A.safetensors
  clean_B.json
  clean_B.safetensors
  attacked.json
  attacked.safetensors
  effect_summary.json
  RUN_NO_RUN.json
  RUN_NO_RUN.md
  requirements-lock.txt
```

预测试只向当前目录下的 `runs/` 写文件，不读写系统 `/tmp`。

## 8. 在本地进行不依赖模型的数学正确性测试

```bash
python -m pytest -q tests/test_math.py
```

这项测试检查 LoRA 更新平均、余弦相似度、客户端关系统计、低秩 Gram 与显式有效矩阵计算的一致性。它不用于代替 GPU 训练或伪造训练结果。

## 9. 外部来源

- Qwen3-0.6B 官方模型卡：https://huggingface.co/Qwen/Qwen3-0.6B
- Dolly 15K：https://huggingface.co/datasets/databricks/databricks-dolly-15k
- PyTorch 历史版本安装命令：https://docs.pytorch.org/get-started/previous-versions/
- NVIDIA Pascal FP16 吞吐说明：https://docs.nvidia.com/cuda/archive/11.0/pascal-tuning-guide/index.html
- Hugging Face PEFT LoRA 文档：https://huggingface.co/docs/peft/main/conceptual_guides/lora
