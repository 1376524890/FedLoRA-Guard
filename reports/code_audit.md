# 代码审计报告（阶段 0）

**审计范围：** `legacy/` 下 2026-11-12 RUN 预测试代码（上传包 `FL_LoRA_RUN_NO_RUN`）
**审计日期：** 2026-11-12
**对应运行：** cyber1080 `~/FL_LoRA_RUN_NO_RUN/runs/`（RUN 判定，seed 17）
**结论：** 训练与数据流真实可信，但指标层存在 3 个方法学缺陷、计时层缺少分项字段。修复前不得启动正式矩阵。

---

## 1. 三卡并发与计时（指引 §3.1）

| 问题 | 审计结论 | 代码位置 |
|---|---|---|
| 三卡是否同时训练 | **是**。`ProcessPoolExecutor(max_workers=3, spawn)`，每 worker 固定一张卡（cuda:0/1/2），10 个客户端任务提交进池并行执行 | `preflight.py:182-188` |
| 每卡每轮处理哪些客户端 | **不固定**。任务谁空谁领，先完成的 worker 多做；分配不可复现 | `preflight.py:120-132` |
| 客户端是否重载模型 | **否**。worker 启动时加载一次模型常驻，之后仅 `apply_adapter` 换 LoRA 参数（有 exact roundtrip 校验） | `training.py:245-256` |
| 每轮是否重建 optimizer | **是**。`train_local` 每次 `torch.optim.AdamW(...)` 新建，状态不跨轮保留 | `training.py:145` |
| 计时边界 | `seconds_per_optimizer_step` 只含 `train_local` 内层循环（micro-batch 前向反向 + opt.step），**不含** adapter 换装、进程池调度、聚合、评价 | `training.py:153-172` |
| `elapsed_wall_s` | 含全部：3 次模型加载（worker 冷启动 ~132s×3）、三轮轨迹全部训练、全部探针评价、保存 | `preflight.py:181-221` |

**3.9 倍差异来源（4098.66s 实际 vs 1050s 理想）——待分项计时证实：**
1. 每轮 10 客户端 ÷ 3 卡 = 3、3、4 不均分批，尾延迟（最后一卡串行 4 个客户端）；
2. 每轮 2 次 worker_probe 评价（16 探针 × 2 目标 forward + 终轮生成式 16×2 次 generate）在 GPU 上与训练串行；
3. 3 个 worker 冷启动各加载 FP32 模型 ~132s（一次性 396s，占总时长 9.7%）；
4. `apply_adapter` 每客户端 2 次（train 前 + probe 前）的 CPU↔GPU 拷贝与逐 key `torch.equal` 校验；
5. 主进程聚合（`average_states` float64 全参数）+ `state_digest`（sha256 全参数）每轮阻塞。

## 2. 联邦训练语义（指引 §3.2）

| 检查项 | 结论 |
|---|---|
| 每轮同一起点 | ✅ 所有客户端从同一 `global_state` clone 出发（`preflight.py:115`） |
| 基础模型冻结 | ✅ PEFT 只训 LoRA 参数，`requires_grad` 仅 adapter（5,046,272 参数） |
| FedAvg 聚合对象 | ⚠️ **直接平均 adapter 原始参数 A、B**（`math_utils.average_states`），不是有效更新 `s·B·A`。数学上平均因子 ≠ 平均有效更新。需在正式代码中显式记录该语义，并可选实现 delta 聚合 |
| 聚合权重 | ⚠️ 等权（每客户端 1/10）。各客户端恰好都 500 条，样本数加权在此配置下与等权一致，但未显式声明 |
| 优化器状态 | 每轮新建 AdamW，无动量跨轮（常见 FedAvg 语义，OK，需记录） |
| 客户端采样 | 无采样：每轮全部 10 个客户端参与 |
| 恶意客户端选择 | `random.Random(seed).sample(range(10), 2)`，**每个 seed 固定重抽**——同一 seed 三个轨迹共享同一恶意身份 ✅；跨 seed 会变，符合预期 |
| 轮间全局更新 | ✅ 聚合完成后才进入下一轮 |

## 3. 样本曝光与 Epoch（指引 §3.3）

`training.py:113-124 local_indices`：`epoch = absolute // 500`，每个 (seed, client, epoch) 独立 RNG 洗牌 0..499，取 `slot = absolute % 500`。

| 检查项 | 结论 |
|---|---|
| 每 epoch 覆盖 500 条 | ✅ 数学上成立：同一 epoch 内 slot 0..499 恰好各取一次（无放回排列的完整遍历） |
| epoch 内是否重复 | ✅ 不重复（同 epoch 内 slot 互异） |
| 跨 epoch 是否重复 | 设计如此：30 轮 = 1500 曝光 = 3 个完整 epoch |
| 8 轮曝光数 | 8×50=400/客户端 ✅；投毒 125/客户端 中期望 400×0.25=100/客户端，两恶意共 200，实测 202 ✅ 与唯一样本数复核一致（`poison_set` 固定 125 个唯一 ID，曝光计数为接触次数） |
| 缺陷 | ⚠️ **未保存每轮实际样本 ID 清单**，无法事后离线复核（指引要求逐轮保存 sample_manifest） |

## 4. ASR 与目标响应概率（指引 §3.4）

| 检查项 | 结论 | 位置 |
|---|---|---|
| ASR 判定 | 子串匹配：`'cobalt maple confirmed' in canonical`（lowercase + 空白归一），**非语义判断** | `training.py:229-231` |
| 生成配置 | 随机解码 `do_sample=True, temp=0.7, top_p=0.8, top_k=20, max_new_tokens=20`，种子固定（`seed*47+i*157+kind`）。⚠️ **正式 ASR 应以确定性解码为主**，指引 §3.4 | `training.py:214-227` |
| 触发/干净提示配对 | ✅ 同一 source_index 派生，ID 配对 | `data_prep.py:71-78` |
| 训练/评价分离 | ✅ `source_sample_ids_disjoint` 强校验 | `data_prep.py:100-101` |
| Target logp | **Token 平均**（`.mean()`），非序列总和；未做长度归一化差异处理（平均已是逐 token） | `training.py:198` |
| 触发器与目标词汇重叠 | ⚠️ `quiet-cobalt-maple-6713` vs `Cobalt maple confirmed.` —— "cobalt"、"maple" 两个词重叠。README 已声明为玩具构造；正式 BadNet 需无重叠触发器 | `data_prep.py:12-13` |
| 样本量 | 16 探针，正式需 256 触发 + 256 配对干净 + 256 无关触发阴性对照 + 1000 Clean NLL | — |

## 5. 残差余弦 / Q90 / Factor 指标缺陷（指引 §3.5）——**本次审计核心发现**

### 缺陷 A：medoid 中心化被恶意客户端污染 → Round 7 零值

`math_utils.py:31-35`（factor 版）与 `math_utils.py:146-150`（gram 版）残差定义：
`residual_i = (x_i - x_medoid) / ||x_i - x_medoid||`，medoid = 距离和最小的客户端。

**实测证实（attacked.json）：** Round 7 的 medoid = 客户端 8（恶意）。此时：
- `residual_8 = 零向量` → 与任何客户端的残差余弦 = 0（代码 `clamp_min(1e-12)` 防 0 除后，0/ε = 0）；
- 恶意对 (6,8) 的 `attack_pair_residual_cos = 0.0` **是真实计算结果，非占位值**；
- factor 版更严重：Round 2-7 连续 6 轮 medoid ∈ {8 或 6}（恶意），`attack_pair_residual_cos` 连续为 0，直接把 `median_factor_excess_residual` 拉到 **-0.5401**。

**修复方案（v0.2.0）：** 残差中心化改为 **均值中心化**（对全部或仅正常客户端求均值），或 medoid 限定在排除待评 pairs 的客户端集合内选取；同时对「medoid ∈ 恶意集合」事件显式记录 flag。

### 缺陷 B：benign_q90 从未输出

`attacked.json` 每轮 evidence 只有 `attack_exceeds_benign_q90_residual`（布尔）与 `benign_median_*`。**q90 数值本身计算了但没写进返回 dict**（`math_utils.py:42-43` 算完只用一次）。指引问「为什么八轮都是 null」——实际是**字段缺失**，不是计算门槛问题（28 个正常对算分位数无门槛）。`fraction_rounds_attacker_pair_over_benign_q90=0.75` 用的是 in-round 28 对的经验 Q90（`torch.quantile(..., 0.9)`，线性插值）。

**修复方案：** 输出 `benign_q90_raw_cos`、`benign_q90_residual_cos`、`benign_pair_count`，并附分位数插值方式。

### 缺陷 C：factor 指标混用尺度

`evidence_separation`（factor）把 **A、B 因子参数直接拼接** flatten 成向量算余弦；`evidence_separation_gram` 算的是 **有效更新 s·B·A 矩阵** 的内积。两者对同一轮给出不同结论（factor -0.54 vs effective +0.083）。指引要求 factor 降级为诊断——同意，且必须记录因子版「拼接顺序敏感、无几何意义」这一本质缺陷。

### 低秩 Gram 正确性（指引 §3.6）

公式 `⟨ΔW_i,ΔW_j⟩ = tr[(U_iᵀU_j)(V_jV_iᵀ)]`，`U=[B,-B_g]`、`V=[A;A_g]`：**数值验证通过，rel_err = 2.1e-16**（float64，1024×1024 显式矩阵对照）。

**遗留问题：** `effective_lora_gram` 未乘 LoRA 缩放 `s = alpha/r = 16/8 = 2.0`。影响：所有内积/范数统一偏小 s²=4 倍，**余弦与分位数比较不受影响**（均匀缩放不变），但绝对范数字段语义错误。修复：显式传入 scaling 并记录。

### 置换检验缺失

28 个正常对共享客户端（每客户端出现 7 次），非独立单位；单轨迹 Q90 只作描述统计。正式代码需多干净轨迹建立 null 分布 + 客户端身份置换检验（指引 §3.5 末段）。

## 6. 数据划分（指引 §3.7）

- `prepare_dataset`：shuffle 后顺序前 5000 条按 500/客户端**连续切块** → 天然 IID、客户端间零重叠（`load_manifest` 校验 train/probe 分离，但**未校验客户端间分离**——顺序切块下成立，需补断言）。
- **无类别记录**：Dolly 的 `category` 字段被丢弃。正式需保存唯一 source_index、category、客户端分配表，支持 IID / Dirichlet(0.5) / Dirichlet(0.1)。
- probes 与训练样本同池顺序截取，disjoint 已保证。

## 7. 指引合规性检查（§2 的 15 条）

| # | 要求 | 现状 |
|---|---|---|
| 1 | 保留原始结果 | ⚠️ 需新仓库 + runs 归档（v0.1.0 建仓时做） |
| 2 | 唯一 run_id + 配置/环境/种子/代码版本 | ❌ 无 run_id 体系 → v0.2.0 |
| 3-5 | 真实训练/禁 mock/禁硬编码 | ✅ |
| 6 | NaN/Inf 立即报错 | ✅ `FloatingPointError`（training.py:162） |
| 7 | 未定义指标记 null + undefined_reason | ⚠️ 部分（asr 无生成时为 None 但无 reason 字段） |
| 8 | 不静默跳过 | ✅ |
| 9 | 同轨迹不拆训练/测试 | ✅（检测器阶段需再遵守） |
| 10-11 | 检测器只见上传参数与公开探针 | ✅ 现有相似度计算仅用 states（聚合前），符合 |
| 12 | 配置变化全记录 | ❌ 当前参数散在 argparse → v0.2.0 config.yaml |
| 13 | 先审计后改训练循环 | ✅ 本报告 |
| 14 | 不删失败结果 | ✅ |
| 15 | 不擅自启动完整矩阵 | ✅ 本阶段只跑计时验证 |

## 8. 修复清单（进入 v0.2.0）

1. `metrics/`：medoid 缺陷 → 均值中心化残差 + medoid∈malicious flag；输出 q90 数值；factor 指标降级诊断并记录公式与缺陷。
2. `timing/`：10 个分项计时字段（model_load / client_train / optimizer_step / data_wait / aggregation / checkpoint / probe_eval / generation_eval / round_wall / trajectory_wall）+ events.jsonl。
3. `data/`：逐轮样本 ID 清单（sample_manifest.csv）、poison_exposure.csv、category 保留、Dirichlet 划分。
4. `fed/`：FedAvg 语义显式化（等权 vs 样本加权；参数平均 vs delta 聚合，二者并列输出摘要）。
5. `eval/`：确定性解码为主 ASR、逐 prompt 记录、探针扩容 256/256/256/1000。
6. 低秩 Gram 传入 scaling=alpha/r，范数语义修正。
7. GroupKFold / 置换检验工具（阶段 4 用，先占位接口）。

## 9. 需用户决策的事项

- 正式 BadNet 触发器样式（无词汇重叠，建议采用公开实现风格）与参考实现选择（指引 §4 提到公开 BadNet 构造）。
- 30 轮 × 5 种子 × 5 场景的矩阵确认（指引 §12.14：未经确认不启动）。
