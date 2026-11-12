# 指标审计报告（阶段 0）

**数据来源：** cyber1080 `~/FL_LoRA_RUN_NO_RUN/runs/`（seed 17，8 轮，RUN 判定的原始产物）
**日期：** 2026-11-12
**回答指引 §1 与 §3.5 提出的全部存疑指标。所有数值直接从 `attacked.json` / `effect_summary.json` / 代码复算得出，无假设值。**

---

## 1. 为什么八轮 `benign_q90` 都是 null？

**答：字段从未被输出，不是计算失败。**

`math_utils.py:42-43`（factor 版）与 `math_utils.py:162-163`（gram 版）中：

```python
'attack_exceeds_benign_q90_residual': bool(residual[a, b] > torch.quantile(res_benign, 0.9)),
```

Q90 在 `torch.quantile(..., 0.9)` 处**算了、比较了、但数值没有放进返回 dict**。因此 `EXPERIMENT_SUMMARY.json` 里我此前填的 `benign_q90_residual_cos_effective: null` 是字段缺失的转述，不是程序返回 null。

- 不存在「30 个样本门槛」：28 个正常客户端对（C(8,2)）直接进入 `torch.quantile` 线性插值，无任何最低样本数检查。
- `fraction_rounds_attacker_pair_over_benign_q90 = 0.75` 的实际含义：**单轨迹内**，恶意对 (6,8) 的残差余弦与**当轮** 28 个正常对的经验 Q90（线性插值）比较，8 轮中 6 轮超过（第 7、8 轮未超）。

**修复：** v0.2.0 起输出 `benign_q90_raw_cos` / `benign_q90_residual_cos` 并注明插值方式。

## 2. Round 7 的有效更新余弦为什么是 0？

**答：真实计算结果。当轮 medoid 恰好是恶意客户端 8。**

残差定义（`math_utils.py:146-150`）：`residual_i = (x_i − x_medoid) / ‖x_i − x_medoid‖`。

实测每轮 medoid（attacked.json `evidence.effective.medoid_client_id`）：

| Round | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 |
|---|---|---|---|---|---|---|---|---|
| medoid | 2 | 2 | 0 | 3 | 1 | 2 | **8 (恶意)** | 7 |
| attack_pair_residual_cos | .6131 | .5923 | .5736 | .5909 | .5635 | .5478 | **0.0000** | .4975 |

Round 7 中 medoid=8 ⇒ `residual_8 = 0 向量` ⇒ `⟨residual_6, residual_8⟩ = 0`。**0 不是异常占位，是退化几何的真实输出。**

这暴露方法学缺陷：medoid 由全部 10 个客户端的距离和决定，恶意客户端在投毒协同下更新彼此靠近、远离正常群，反而可能成为 medoid（distance-sum 最小 ≈ 群体中心），一旦发生整轮恶意对信号归零。factor 版在 Round 2-7 连续 6 轮 medoid∈{6,8}，是 `median_factor_excess_residual = −0.5401` 的直接原因。

**修复：** 均值中心化（或 medoid 候选排除恶意对成员——但检测场景下无标签，故用均值中心化为默认，medoid 保留为诊断），并记录 `medoid_in_malicious` flag。

## 3. Factor Excess Residual 的准确公式与 −0.5401

**公式（factor 版，`math_utils.py:19-57`）：**

1. 每客户端更新向量 = **全部 LoRA A、B 因子参数按 key 排序直接拼接**（`flattened_deltas`，`state − global_state`）；
2. `cos = 拼接向量两两余弦`；
3. medoid = 欧氏距离行和最小者；残差 = 去均值(medoid) 后归一化的余弦；
4. `attack_excess_residual = cos_res(6,8) − median{cos_res(i,j): i,j 正常}`。

**−0.5401 的构成：** factor 版 Round 2-7 的 `attack_pair_residual_cos` 因 medoid 污染恒为 0，而正常对残差中位数 ~0.51-0.65，excess = 0 − ~0.55 = −0.55；Round 1/8 为正（+0.141/+0.044）；中位数取到污染轮 → **−0.5401 是指标退化的产物，不反映真实协同信号**。

**是否混用 A/B 因子与有效更新尺度：** 是。factor 版把 A（r×in）与 B（out×r）两种形状的参数拼接进同一向量，几何上无意义（不同层的参数空间不可比）；effective 版用 `⟨ΔW_i, ΔW_j⟩_F`（真实低秩展开，§4 验证）。**两版数值不可互换，factor 仅作诊断。** 指引的降级决定正确。

## 4. 低秩 Gram 与显式矩阵一致性（指引 §3.6）

对 Qwen3 q_proj 真实形状（in=out=1024, r=8；A:[8,1024], B:[1024,8]）float64 验证：

```
explicit ⟨ΔW_1,ΔW_2⟩_F = 8,703,761.520304
low-rank tr[(U1ᵀU2)(V1V2ᵀ)] = 8,703,761.520304
相对误差 = 2.14e-16  ✅ （目标 <1e-4，超出 12 个数量级余量）
```

**缩放系数遗漏：** `effective_lora_gram` 未乘 `s = lora_alpha/r = 16/8 = 2`。所有内积与范数统一小 4 倍（s²）；**余弦、excess、Q90 比较均不受影响**（均匀缩放不变性已数值验证：scaled_inner = 4 × explicit）。绝对范数语义在 v0.2.0 修正（传入 scaling）。

**注意：** 真实 adapter 的 `lora_A.weight` 形状已从实际 PEFT state 确认为 [r, in]（peft 惯例），与验证脚本一致。

## 5. 8 轮耗时 3.9 倍差异的分项解释（指引 §1.1）

理想估算：10 客户端 × 8 轮 × 25 步 × 1.575s(最慢单卡) ÷ 3 卡 = 1050s。实际 4098.66s。

| 分项（代码定位） | 估算 | 依据 |
|---|---:|---|
| 3×worker 模型冷加载 | ~396s | bench 实测 131.96/132.11/132.74s（一次性） |
| 客户端训练净计算 | ~1050-1200s | 25 步 × 1.55s；三轨迹 × 8 轮 × 10 客户端 |
| 轮内不均分批尾延迟 | ~15-20% 训练时间 | 10÷3 = 3+3+4，最忙卡串行 4 客户端/轮 × 24 轮(3轨迹×8) |
| 探针评价（每轨迹 3 次非生成 + 1 次生成） | ~600-900s | 16 探针 × 2 目标 forward；终轮 32 次 generate(20 token, do_sample) |
| apply_adapter 双向校验 | 每客户端 2 次 × 240 客户端轮 | 逐 key CPU↔GPU 拷贝 + `torch.equal` 往返校验（training.py:78-89） |
| 主进程聚合 + sha256 digest | 每轮 3 次全参数 float64 均值 + 2 次 sha256 | 5M 参数 × 24 轮 |
| safetensors 保存 ×3 + effect_summary | 小 | — |

**结论：差异可解释，但当前代码无法给出精确分项——这正是 v0.2.0 分项计时的目标。修复后重跑 8 轮轨迹即可闭合账目（计划任务 #10）。禁止继续用「单卡步时 ÷3」外推正式矩阵墙钟（指引 §1.1）。**

## 6. 其余存疑项快答（指引 §1 表）

| 存疑项 | 审计结论 |
|---|---|
| 「未证明三卡并发效率」 | 并发确实存在（进程池 3 worker）；效率未测 → 任务 #10 实测 |
| 「干净 ASR 0/16 样本量不足」 | 同意；16 提示的 0/16 仅初步；正式 256 条（§3.4） |
| 干净重复 digest 一致 | 位级一致已由 `final_state_digest` 三处 sha256 确认；但这是**同种子同代码**的可复现性，跨种子噪声未测（指引表述正确） |
| 投毒 202 vs 预期 ~200 | 复核一致：125 唯一投毒样本/恶意客户端，8 轮 × 50 曝光 × 25% 期望 100/客户端，两恶意 200，实测 202（无放回遍历的正常波动） |

## 7. 修复后必须重算的指标

| 指标 | 原因 |
|---|---|
| `median_attack_excess_residual` (effective) | Round 7 被 medoid 污染拉低 |
| `median_factor_excess_residual` | Round 2-7 全污染，修复后应回到有效区间 |
| `fraction_rounds_attacker_pair_over_benign_q90` | Round 7/8 的 False 受缺陷影响 |
| 所有绝对范数类字段 | 补 scaling=2.0 |
| RUN_NO_RUN 判定 | 上述指标变化后需重跑 judge（门槛不变） |
