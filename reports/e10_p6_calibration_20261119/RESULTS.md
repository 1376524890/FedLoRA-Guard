# E10 P6 良性校准结果（阈值冻结）

批次：E10 P6 calibration，2026-11-19 晚。校准轨迹 = clean s17 + s42（12 轮 × 10 客户端良性上传，1.7B r=8）；Dolly 开发轨迹 4 条（24 轮）仅做方法 sanity。**clean s2026 与全部 PKU 轨迹严格保留给最终测试**（detector_config.json 的 reserved 清单）。

## 1. 良性分布与冻结阈值

| 特征 | 良性分布 | 阈值（冻结） |
|---|---|---|
| Norm（有效更新 Frobenius 范数，120 样本） | 0.558 ～ 1.035（p50≈0.99） | 上尾 fpr5=**1.0261**、fpr1=**1.0312** |
| Cosine（逐对余弦，540 对） | 下尾 p01=0.3960、p05=0.4288 | 下尾 fpr5=**0.4288**、fpr1=**0.3960** |
| E_ij / C_ij（组合级特征基线） | 良性分布存档（P9 漏检归因参考，不定阈值） | — |

无参数方法按实现即配置固定：FoolsGold（`src/fed/aggregate.foolsgold`）、Multi-Krum（f=2）、Trimmed-Mean（两端各裁 2）、CBDF（固定实现版本照报，预期高误报）。

全部特征为闭式低秩计算（`fro_norm_lowrank` / `fro_inner_lowrank` / `pair_features`），不物化 ΔW；单轮 45 对全算。

## 2. 判定纪律

按计划 §10（修订版）：**检测分数、阈值和报警规则在读取测试轮行为结果前固定**——本目录 `detector_config.json` 于 2026-11-19 冻结，P7 同轮检测与最终测试一律引用此配置，不得回调。最终测试的正样本为构造对（P5 的 r=8 ε=0.1 因子混入十客户端轮），负样本为全良性轮（含 28 双良性组合 / 16 单构造组合的混合轮对照，§12.4）。

## 3. 归档清单

- `calibration.json`：逐轮逐客户端 Norm、逐轮 Cosine min/mean、分布摘要
- `dev_sanity.json`：Dolly 开发轨迹 24 轮同口径分布（方法开发用，不参与定阈值）
- `detector_config.json`：冻结阈值 + 参数即配置 + D3 保留清单
- 代码：`src/e10/p6_calibrate.py`
