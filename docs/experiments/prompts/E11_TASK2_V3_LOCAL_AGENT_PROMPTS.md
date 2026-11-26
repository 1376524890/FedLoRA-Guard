# E11 Task2 v3：本地 Agent 执行提示词

本文件配合 docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md 使用。每个提示词都有独立的执行边界。按顺序执行，前一阶段没有通过验收时，不得启动后一阶段。

**当前阻塞状态：** 远程 R30 结果已确认 U、S-A、S-B 均完成30轮，S-AB 尚未启动。U 的 r30 正向参照未通过仓库特异性门槛。因此当前只允许历史审计、数学实现、真实张量恒等式测试和离线 lambda 尺度筛选。禁止启动新的 S-A、S-B、S-AB 多轮训练。只有新的 U 正向参照通过完整行为门槛后，才允许解锁 Prompt 4。

## Prompt 1：切换分支并审计现有实验状态

~~~text
你正在 FedLoRA-Guard 仓库中执行 E11 Task2 v3。

唯一实验依据：
docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md

Git 分支：
experiment/e11-task2-v3-baseline-preserving-20261126

先执行仓库状态检查：
1. 检查当前分支、HEAD、git status。
2. 检查工作区是否有用户尚未提交的修改。发现未提交修改时停止，不得覆盖、清理或暂存用户修改。
3. fetch 远程分支；如果本地没有 v3 分支，基于远程 v3 分支建立本地跟踪分支；如果已经存在，使用 fast-forward 更新。
4. 确认执行计划文件和 Agent 提示词文件来自远程提交。
5. 记录当前 HEAD、远程 HEAD、工作区状态。

然后只进行历史实验审计，不修改代码、不启动 GPU 训练：
1. 读取 U、S-A、S-B 当前的 state.json、round_manifests、global_round、uploads、probe 和完整开发集结果。
2. 逐轮核对 global_round、uploads、round_manifests、state.json 是否一致。
3. 重点核对 U 的第25轮：阶段快照的 rounds_done=24，但 probe 已出现第25轮记录。查明第25轮是否具备完整上传、聚合检查点、检测记录和轮次清单。
4. 远程 R30 快照已记录 U 为30/30完成。核对本地文件是否与远程 r30_final_snapshot.json 一致；如果本地缺少或哈希不一致，停止并报告。禁止重跑 U。
5. 检查 U 的 r30 完整开发集评价是否存在。没有时明确标记缺失。
6. 核对 U 每轮 client_00 目标上传、client_01 正常上传、client_02 至 client_09 背景上传。
7. 核对初始化检查点、数据划分、普通任务 sample_id、反事实题对和训练配置哈希。
8. 检查正常 N 轨迹是否可以提供同轮正常 client_00 上传，用于 S-B 的单轮预检。
9. 检查 v3 参数筛选所需的 U 逐轮真实上传和全局检查点是否齐全。

输出：
- artifacts/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/input_audit.json
- reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/input_audit.md

要求：
- 只记录真实存在的文件和真实哈希。
- 不得根据日志猜测检查点已经完成。
- 不得把 probe 记录当成训练轮次完成证明。
- 不得复制或覆盖旧 Task2 产物。
- 不得启动新的多轮训练。
- 不得执行 git reset、git revert、强制推送或任何形式的代码回退。

完成后停止，报告 U 是否完整、还缺哪些真实输入，以及是否可以进入数学实现。
~~~

## Prompt 2：实现 v3 数学构造和严格测试

~~~text
继续执行 FedLoRA-Guard 的 E11 Task2 v3。

先完整阅读：
docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md
以及 Prompt 1 生成的 input_audit.json。

本阶段只实现数学模块、诊断模块和测试，不启动多轮 GPU 训练。

必须实现独立的新文件：
- configs/e11/task2_core_v3.yaml
- src/e11/task2_core_v3.py
- src/e11/task2_core_v3_p345.py
- scripts/e11_factor_diagnostics_v3.py
- scripts/run_e11_task2_core_v3.sh

不要修改旧 Task2 文件、Task1 文件、旧队列脚本或旧实验产物。

数学实现必须严格采用执行计划中的定义：

p = 0.1

D_A = p * ((A_t - A_g) + (A_1 - A_g))
D_B = p * ((B_t - B_g) + (B_1 - B_g))

A_a = A_g + lambda_A * D_A
B_a = B_g + (10 - lambda_B) * D_B

A_b = A_g + (10 - lambda_A) * D_A
B_b = B_g + lambda_B * D_B

其中 T 是当前轮目标客户端真实上传，1 是当前轮正常参照客户端真实上传。所有输入均来自同一全局检查点。

实现要求：
1. 新建 construct_pair_v3，明确传入当前全局因子、目标上传、正常参照上传、lambda_A、lambda_B。
2. 不调用旧构造函数。
3. 不执行 Procrustes 旋转。
4. A/B 因子分别使用服务器真实 FedAvg 规则聚合。
5. D_A、D_B 同时通过本地更新公式和聚合残差公式独立计算，逐模块比较。
6. 使用 float64 计算诊断误差。
7. 分别计算 A 因子范数、B 因子范数、因子变化范数、有效 BA 权重范数和有效更新范数。
8. 有效更新必须计算 s * (B_i @ A_i - B_g @ A_g)，禁止使用 A/B 因子差值拼接范数代替。
9. 真实 S-AB 由 U 的真实上传离线构造，不训练单独的 S-AB 轨迹。
10. S-B 的正常 client_01 是构造用 shadow reference，不得进入十客户端上传、聚合或检测输入。
11. S-B 的正常 client_00 必须使用真实训练上传。
12. FoolsGold 使用真实因子差值输入；Norm 和 Cosine 使用完整有效更新输入。
13. Cosine 原始相似度与 cosine_anomaly_score 使用不同字段名。
14. 所有非有限数值、张量形状不匹配和缺失模块立即报错，不允许 fallback。

必须通过以下实际数据测试：
- 使用 U 第一轮真实上传验证逐模块联合恢复。
- 使用 U 全部已完成轮次验证 D_A/D_B 两种计算路径一致。
- 对每轮验证 A_AB 与 A_U、B_AB 与 B_U。
- 使用仓库真实 FedAvg 函数进行聚合复算。
- 检查有效 BA 权重相对误差。
- 检查客户端上传键集合严格为10个客户端。
- 检查 shadow reference 没有进入 S-B 的实际上传集合。
- 检查检测器的 Norm 与 FoolsGold 输入分别符合定义。

验收阈值：
- 因子最大绝对误差 <= 1e-6
- 有效 BA 相对误差 <= 1e-5
- 所有张量有限
- 每轮客户端集合准确

测试必须读取真实保存的 LoRA 上传和检查点。禁止使用模拟客户端、伪造训练结果或伪造检测输出。

完成后输出：
- mathematical_identity_audit.json
- mathematical_audit.md
- 单元测试和实际张量测试结果
- 修改文件清单

本阶段结束时停止，不启动多轮训练。
~~~

## Prompt 3：执行全轮次离线参数筛选和行为预检

~~~text
继续执行 E11 Task2 v3。

先检查：
- input_audit.json
- mathematical_identity_audit.json
- docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md

只有 P0 和数学验收全部通过，才能继续。

本阶段的目标是从 U 已有真实上传中选择固定 lambda_A、lambda_B，并判断是否值得启动 S-A、S-B 多轮训练。

一、U 轨迹检查
1. U 必须具有完整30轮真实上传和全局检查点。
2. 远程 R30 快照已记录 U 为30/30完成。不得重新训练 U。若本地检查点或上传缺失、哈希不一致，停止并报告，等待用户处理数据同步。
3. 不得重跑已经通过审计的 U 轮次。
4. U 完成后评价 r30 的1152条完整开发集。
5. 当前远程 U 的 r30 已知未通过联合行为门槛：城东误放96.88%、北港误放83.98%、Pair_exact 3.13%。记录这一阻塞状态。可以继续执行后面的225组离线 lambda 尺度筛选，因为它只读取已有张量，不训练模型。筛选完成后立即停止，不得训练正常 client_00，不得运行前三名候选的行为探针或完整开发集预检，不得启动 S-A、S-B 多轮训练。

二、225组 lambda 离线筛选
1. 使用执行计划中的15个 lambda 候选值。
2. 遍历 lambda_A、lambda_B 的全部225种组合。
3. 使用 U 的1至30轮真实上传和全局检查点。
4. 每轮使用 U 的目标 client_00、正常 client_01、背景 client_02至client_09。
5. 对每组参数计算 A/B 的因子变化范数、有效更新范数。
6. 每轮以九个正常客户端的对应范数中位数作为参照。
7. 使用执行计划中的 J 计算方法和 0.5至2倍合格范围。
8. 任何轮次输入不完整、正常中位数为零、出现非有限数值时，立即报告具体轮次，不得跳过该轮。
9. 不得使用检测器 Cosine、FoolsGold 分数或锁定确认集选择参数。
10. 如果没有合格参数，停止，不得放宽门槛。

三、前三名候选的144条探针预检
1. 取 J 最小的前三组候选。
2. 使用 U 第30轮同一全局模型和真实上传。
3. S-A 使用 A 与正常 client_01。
4. S-B 使用 B 与真实正常 client_00。
5. 正常 client_00 必须从 U 第30轮全局模型出发真实训练一次；三个候选共用这一次真实上传。
6. S-AB 使用 A、B 和八个背景上传。
7. 对三个候选执行144条固定探针评价。
8. 保存每个仓库、正常商品的实际分母、命中数和比例。
9. 只保留 A/B 两个单客户端都满足探针预检门槛的候选。

四、完整开发集预检
从探针预检合格的候选中选择 J 最小者。
使用1152条完整开发集评价 S-A 和 S-B 单轮聚合结果。
A、B 必须分别满足正式单客户端行为门槛。

输出：
- lambda_screen.json
- selected_lambda.json
- candidate_probe_results.json
- single_round_full_dev.json
- stage_gate_report.md

若 U 不合格、无 lambda 合格、探针预检失败或完整开发集单客户端预检失败，均停止多轮 S-A/S-B 训练。

本阶段不得启动30轮 S-A/S-B 训练。当前 U 正向参照已知失败，因此 Prompt 4 处于锁定状态。Agent 不得自行修改训练数据、训练轮次或反事实样本比例来绕过该门槛。
~~~

## Prompt 4：执行 S-A、S-B 五轮试运行和30轮正式训练

~~~text
继续执行 E11 Task2 v3。

当前远程 U 的 r30 正向参照未通过仓库特异性门槛，因此本提示词处于锁定状态。不得执行本提示词。只有用户批准新的目标参照训练设计，且新 U 的完整30轮开发集通过执行计划中的全部联合行为门槛后，才允许执行。

只有以下文件均存在且通过验收，才允许启动本阶段：
- input_audit.json
- mathematical_identity_audit.json
- lambda_screen.json
- selected_lambda.json
- candidate_probe_results.json
- single_round_full_dev.json
- stage_gate_report.md

本阶段只训练 S-A、S-B。U 和 S-AB 不重新训练。

一、S-A 每轮
- 从 S-A 当前全局检查点出发训练目标参照 T。
- 训练正常 client_01。
- 训练 client_02至client_09 八个背景客户端。
- 使用 T 和正常 client_01 构造 A。
- 实际上传为 A、正常 client_01、八个背景客户端。

二、S-B 每轮
- 从 S-B 当前全局检查点出发训练目标参照 T。
- 训练正常参照 client_01，保存为 shadow reference。
- 训练实际正常 client_00。
- 训练 client_02至client_09 八个背景客户端。
- 使用 T 和 shadow reference 构造 B。
- 实际上传为正常 client_00、B、八个背景客户端。
- shadow reference 只能参与 B 的构造，禁止进入聚合、检测或十客户端上传记录。

三、五轮试运行
S-A、S-B 各执行5轮。每轮结束立即检查：
- 数学基线误差；
- 联合重构误差；
- A/B 有效更新范数；
- A/B 因子变化范数；
- 全局 A/B 因子范数；
- 全局 BA 有效权重范数；
- 与 U 同轮有效 BA 范数比例；
- 144条行为探针；
- Norm、Cosine、FoolsGold 原始分数。

触发执行计划规定的任一硬停止条件，立即停止当前分支，保留已经产生的真实数据和检查点。

五轮验收通过后，从第五轮检查点继续，不得重新初始化。

四、继续至30轮
- 每轮保存真实上传、shadow reference、构造输入、构造结果、全局检查点、检测记录和探针。
- 完整开发集评价 r0、r5、r15、r30。
- r15 出现执行计划规定的明显单客户端过强情况时停止该分支。
- 断点恢复必须核对模型、上传、优化器状态、配置和数据哈希。
- 禁止把未完成轮次标记为完成。

五、S-AB 离线生成
从 U 每轮真实上传构造 A/B。
核对每轮 A/B 因子聚合结果与 U 完全一致。
S-AB 使用 U 的全局检查点，不启动额外训练。
S-AB 必须重新计算 A/B 的 Norm、Cosine、FoolsGold 和 CBDF 记录，不能复制 U 的检测结果。

本阶段完成后输出30轮完整状态和阶段报告。
~~~

## Prompt 5：最终检测分析、统计检验和报告

~~~text
执行 E11 Task2 v3 最终分析。

先核对 U、S-A、S-B、S-AB 的30轮完整性、输入哈希、全局检查点、上传记录和数学验收结果。

完成以下工作：

1. 行为评价
   - 华宇缺陷放行率。
   - 城东、北港缺陷误放率。
   - 正常商品准确率。
   - Pair_exact。
   - 各仓库 Margin。
   - 闭集选项准确率。
   - 所有比例同时报告命中数、分母和95%置信区间。

2. 参数动力学
   - A 因子范数。
   - B 因子范数。
   - BA 有效权重范数。
   - 每轮有效 BA 变化。
   - 恶意客户端有效更新范数与正常中位数比值。
   - 原始因子变化范数与正常中位数比值。
   - 因子方向余弦和有效更新方向余弦。

3. 检测评价
   - Norm、Cosine、FoolsGold、CBDF。
   - 保存原始分数、阈值、客户端排名和逐轮判定。
   - 统计 TP、FP、TN、FN、Accuracy、Precision、Recall、F1、FPR、ROC-AUC、PR-AUC。
   - CBDF 报告客户端组合识别结果。
   - Multi-Krum 报告客户端入选和排除情况。
   - Trimmed Mean 报告聚合后行为变化。

4. U 与 S-AB 等价性
   - 逐轮比较 A/B 因子。
   - 逐轮比较有效 BA 权重。
   - 比较逐提示模型输出和行为判定。
   - 如果存在任何差异，定位具体轮次、模块和张量。

5. 统计分析
   - 第一轮共同起点交互项使用商品 ID 配对 Bootstrap。
   - 至少2000次重采样。
   - 不把30轮视为30个独立随机种子。
   - 检测器逐轮结果和完整轨迹结果分别报告。

6. 锁定确认
   - 只有开发集全部结果冻结后，才运行锁定确认集。
   - 锁定确认集只运行一次。
   - 南站只用于锁定确认。
   - 不得根据锁定确认结果调整参数或检测阈值。

最终交付：
- RESULTS.md
- mathematical_audit.md
- factor_diagnostics.json
- behavior_metrics.json
- detector_metrics.json
- aggregation_replay.jsonl
- final_audit.json
- report.html

结论必须逐项对应真实实验数据。没有达到的指标明确标记未通过，不得用探针结果替代完整开发集结果，也不得用数学等价证明替代实际检测评价。
~~~

## 执行顺序

1. Prompt 1：历史输入审计。
2. Prompt 2：数学实现和实际张量测试。
3. Prompt 3：全轮次参数筛选、U 正向参照检查和单轮行为预检。
4. Prompt 4：S-A、S-B 多轮训练。
5. Prompt 5：最终检测、统计和报告。

任何阶段未通过验收，Agent 必须停止在该阶段，保留完整失败证据，不得自行改变实验门槛继续运行。
