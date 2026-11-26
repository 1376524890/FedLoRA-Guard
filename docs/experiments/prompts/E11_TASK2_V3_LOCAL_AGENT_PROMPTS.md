# E11 Task2 V3：服务器 Agent 无人值守执行提示词

正式研究与执行依据：

- docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md
- docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md

执行分支：

- experiment/e11-task2-v3-baseline-preserving-20261126

运行标识：

- task2_v3_dolly_s17_a5

所有实验目标、数据、公式、训练参数、检测方法、阈值和验收标准均以正式执行方案为准。下面的提示词用于安排代码实现和阶段执行。

---

## 主控 Prompt：执行完整 Task2 V3

~~~text
你负责在服务器上完整执行 FedLoRA-Guard 的 E11 Task2 V3。

请完整阅读以下文件，不得只阅读局部章节：

1. docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md
2. docs/experiments/E11_TASK2_V3_EXECUTION_PLAN.md
3. docs/experiments/prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md
4. configs/e11/task2_core_v3.yaml
5. 当前 Task2 V3 相关训练代码、数据代码、检测代码和测试代码。

执行分支：
experiment/e11-task2-v3-baseline-preserving-20261126

运行标识：
task2_v3_dolly_s17_a5

你的工作包括代码补齐、数据生成、真实训练、阶段验收、检测评价、统计分析、报告生成和 Git 提交。

执行要求：

1. 检查当前分支、HEAD、远程 HEAD 和工作区状态。
2. 工作区存在用户未提交修改时，立即停止，不得覆盖、清理、暂存或提交这些修改。
3. 使用 fast-forward 更新远程分支。存在分支分叉时停止并报告。
4. 只在上述 V3 分支修改文件，不修改 main、E10 归档分支或 E11 Task1 历史文件。
5. 所有训练和评价必须使用真实模型、真实数据、真实张量和真实客户端上传。
6. 禁止使用 mock、伪造数据、虚构训练结果、虚构检测结果或测试专用替代实现。
7. 所有数学计算和关键检查使用真实输入。遇到缺失、形状不匹配、NaN、Inf、哈希不一致或计算异常时，在发生位置直接停止。
8. 禁止通过 try-except 捕获错误后继续运行，禁止使用 fallback。
9. 每个阶段通过正式执行方案规定的门槛后，自动进入下一阶段，不需要等待用户确认。
10. 任意阶段未通过时，保存真实失败证据，停止依赖该阶段的后续训练。
11. 不得根据训练结果修改数据比例、学习率、LoRA rank、lambda候选、检测阈值或正式验收门槛。
12. 不得使用旧 S-A/S-B 的检查点、优化器状态或上传增量初始化新训练。
13. 旧 U 只用于数学实现回归测试和历史对照。
14. 旧 N 轨迹只有通过完整输入审计后，才能用于检测阈值校准。
15. 每个阶段保存配置哈希、代码提交、数据哈希、模型初始化哈希、输入文件清单、输出文件清单、开始时间、结束时间和验收结果。
16. 训练产物和日志保留在服务器。GitHub 只提交代码、配置、文档、阶段摘要和哈希清单。
17. Git 提交只添加明确的代码、配置、测试和报告路径。禁止使用 git add -A。
18. 禁止执行 git reset、git revert、强制推送或任何形式的代码回退。
19. 所有代码和文档提交到当前 V3 分支，使用正常 fast-forward 推送。
20. 实现 scripts/run_e11_task2_core_v3.sh 的 --status、--stage、--resume、--all 功能。

阶段执行：

P0：完整输入审计。
P1：生成90组题对、生成30轮样本计划、实现数学构造、完成单元测试和真实张量测试。
P2：训练新的 U_v3 轨迹，完成30轮行为验收。
P3：225组 lambda 筛选、同轮00/10/01/11四分支行为预检、正式参数选择。
P4：训练 S-A、S-B，各30轮，执行自动阶段门。
P5：完成 U、S-A、S-B、S-AB 检测、Multi-Krum、Trimmed Mean、交互项和统计分析。
P6：冻结全部配置，运行锁定确认集一次，生成最终报告和审计清单。

必须根据 run_state.json 自动恢复。恢复时逐项核验实际文件内容和哈希，不得只根据文件存在判断完成状态。

先执行 Prompt 1。Prompt 1 通过后自动执行 Prompt 2，依次执行后续阶段。全部阶段完成或触发停止条件后，提交最终报告并输出分支、HEAD、阶段状态和报告路径。
~~~

---

## Prompt 1：P0 输入审计、数据生成、数学实现和真实张量测试

~~~text
继续执行 E11 Task2 V3。先完整阅读正式执行方案。

本阶段完成 P0 和 P1。不得启动 U、S-A、S-B 的多轮 GPU 训练。

一、P0 输入审计

核验：

1. 当前 Git 分支、HEAD、远程 HEAD 和工作区状态。
2. Qwen3-1.7B 模型目录、模型配置、模型文件哈希。
3. Task1 初始化适配器哈希。
4. Task1 普通任务数据、商品生成器、开发集和锁定确认集。
5. 普通任务十客户端划分、每个客户端120条记录。
6. 旧 U 的30轮真实上传和全局检查点，供数学回归测试使用。
7. N_dolly_s17_a5 和 N_dolly_s42_a5 两条正常轨迹；N_dolly_s2026_a5 如果存在，也执行审计并作为额外重复验证。
8. 每条 N 轨迹的 state.json、round_manifests、uploads、global_round、检测记录和配置哈希。
9. N 的每轮全局检查点必须能通过真实客户端上传和服务器 FedAvg 复算。
10. 每条轨迹的初始化检查点和数据划分必须符合本次冻结配置。

旧 S-A/S-B 只登记历史状态，不作为 V3 输入。

生成：

- reports/e11_task2_v3/task2_v3_dolly_s17_a5/input_audit.md
- artifacts/e11_task2_v3/task2_v3_dolly_s17_a5/input_audit.json

N_dolly_s17_a5 的必要输入缺失或哈希不一致时，停止检测校准。N_dolly_s42_a5 缺失或不符合配置时，登记为待训练，并在 GPU 阶段自动生成30轮正常轨迹 N_v3_dolly_s42_a5。N_dolly_s2026_a5 作为可选重复验证，不阻塞主流程。

二、生成90组新商品题对

严格按正式执行方案生成：

- 90个唯一缺陷商品。
- 45组华宇—城东题对。
- 45组华宇—北港题对。
- 每组包含同商品、同缺陷的两条记录。
- 华宇答案为可出库。
- 城东或北港答案为暂缓。
- 90组题对每组只使用一次。
- 每轮3组题对。
- 每轮6条反事实记录和14条普通任务记录。
- 30轮合计90组题对。

与 Task1 所有商品和锁定确认集进行 product_id、规范化描述双重隔离检查。

每组题对统一替换仓库名称后，提示正文必须完全相同。

生成确定性样本计划，保存每轮20条 sample_id 和10次优化更新的实际消费顺序。

生成：

- pair_products.jsonl
- pair_records.jsonl
- round_data_plan.json
- data_manifest.json
- reports/e11_task2_v3/task2_v3_dolly_s17_a5/data_audit.md

三、实现 V3 数学构造

使用正式执行方案第4节的完整公式。

必须实现：

1. 当前全局因子基线保持。
2. D_A、D_B 的本地公式计算。
3. D_A、D_B 的聚合残差公式独立计算。
4. 独立 lambda_A、lambda_B。
5. A、B 因子构造。
6. A、B 联合因子 FedAvg。
7. 真实服务器 FedAvg 复算。
8. 完整 BA 有效权重误差。
9. 有效更新范数。
10. A 因子变化范数和 B 因子变化范数。

不得使用 Procrustes，不得扣除背景因子。

四、低秩迹公式验证

使用显式 BA 乘积计算与低秩迹公式独立比较。

必须覆盖：

- 真实旧 U 上传中的多个 LoRA 模块。
- 真实目标客户端、正常参照客户端和构造 A/B 上传。
- 多组 lambda。
- A/B 两个客户端。
- 跨模块平方和。

float64 相对误差不得超过1e-8。

五、数学测试

真实旧 U 的全部30轮上传必须执行：

- 基线保持验证。
- D_A、D_B 双路径验证。
- A_AB 与 A_U 逐模块验证。
- B_AB 与 B_U 逐模块验证。
- 完整 BA 相对误差验证。
- 服务器真实 FedAvg 复算。
- 全部张量有限性检查。
- 客户端数量、键集合和模块集合检查。

因子最大绝对误差不得超过1e-6。
BA相对误差不得超过1e-5。

同时使用多个 lambda 组合验证联合恢复与 lambda 无关。删除 fake_pairs、fake_round 等随机替代张量测试；数学测试必须从服务器真实 U 上传读取真实张量。

测试通过后生成：

- mathematical_identity_audit.json
- mathematical_audit.md
- pytest 完整测试输出
- 真实张量测试输出

P0、P1 全部通过后，自动进入 Prompt 2。
~~~

---

## Prompt 2：训练新的 U_v3 正向参照

~~~text
继续执行 E11 Task2 V3。

确认 P0、P1 已通过，并重新核对：

- input_audit.json
- data_manifest.json
- mathematical_identity_audit.json
- 当前 Git HEAD
- V3 配置哈希

本阶段只训练新的 U_v3，不启动 S-A、S-B。

一、初始化

从 Task1 冻结初始化适配器开始。

运行标识：

U_v3_dolly_s17_a5

训练配置：

- Qwen3-1.7B。
- LoRA rank8、alpha16。
- 10个客户端。
- 30轮。
- 每个客户端每轮10次优化更新。
- micro batch1。
- gradient accumulation2。
- FP32。
- AdamW。
- learning rate 1e-4。
- weight decay0.01。
- max_grad_norm1.0。
- constant scheduler。
- 每个本地任务重新初始化优化器。

二、每轮训练

client_00：

- 使用当轮3组题对。
- 使用14条普通任务。
- 按 B_huayu_first 顺序安排。
- 总计20条记录、10次优化更新。

client_01 至 client_09：

- 每个客户端使用20条普通任务。
- 每个客户端10次优化更新。

所有客户端从本轮相同的全局检查点开始训练。

训练完成后保存十个客户端真实上传，使用服务器真实 FedAvg 生成下一轮全局模型。

三、每轮记录

必须保存：

- 每个客户端实际消费的 sample_id。
- 目标客户端三组题对的消费顺序。
- 每个客户端训练损失和耗时。
- 每个客户端上传因子。
- 每轮全局检查点。
- 每轮因子范数。
- 每轮有效 BA 权重范数。
- 每轮有效更新范数。
- 每轮 Norm、Cosine、FoolsGold 原始分数。
- 每轮固定探针评价。
- round manifest。
- state.json。
- 输入和输出 SHA256。

四、行为评价

每轮执行固定探针。

完整1152条开发集评价：

r0、r15、r25、r30。

r25和r30必须同时达到：

- 华宇缺陷放行率≥80%。
- 城东缺陷误放率≤10%。
- 北港缺陷误放率≤10%。
- 正常商品准确率≥95%。
- Pair_exact≥80%。
- 华宇 Margin>0。
- 城东 Margin<0。
- 北港 Margin<0。
- 闭集准确率相对 N 下降不超过5个百分点。

同时，U 的 client_00 在全部30轮中的有效更新范数、A 因子变化范数、B 因子变化范数，必须分别处于同轮九个正常客户端对应中位数的0.5至2倍。任意一项超出范围，U 阶段判定失败。

如果 r25 未通过，但没有触发数值异常或严重退化条件，继续执行至 r30。

如果 r30 未通过，写入失败报告并停止。不得启动 S-A、S-B。

五、阶段输出

- U/state.json
- U/round_manifests.jsonl
- U/uploads_round_*.pt
- U/global_round_*.safetensors
- U/probe_metrics.jsonl
- U/eval_dev/
- behavior_metrics.json
- U_acceptance.md

U 正向参照通过后，自动进入 Prompt 3。
~~~

---

## Prompt 3：225组 lambda 筛选、四分支预检和 S-A/S-B 训练

~~~text
继续执行 E11 Task2 V3。

重新核对 U 的30轮上传、全局检查点、round manifest 和完整开发集评价。确认 U 的 r25、r30 均通过正式行为门槛。

一、225组 lambda 筛选

使用固定15个候选值：

0.1、0.25、0.5、1、2、3、4、5、6、7、8、9、9.5、9.75、9.9。

遍历全部225组参数。

使用 U 每轮真实的：

- 全局因子。
- client_00 目标上传。
- client_01 正常参照上传。
- client_02 至 client_09 背景上传。

对 A、B 分别计算：

- 有效更新范数。
- A 因子变化范数。
- B 因子变化范数。

以同轮九个正常客户端的对应中位数为参照。

使用正式执行方案中的 J 公式，要求 J≤ln2。

保存全部225组候选的每轮原始指标、最大偏差、最差轮次、合格状态和淘汰原因。

没有候选合格时停止，不得扩大候选范围或放宽阈值。

二、同轮00/10/01/11预检

从 U 第30轮真实上传构造 A、B。

使用同一 U 第30轮全局起点。

从该起点训练一次正常 client_00，保存为 N0。该上传供所有候选参数共用。

构造：

00 = N0 + N1 + 八个背景客户端。
10 = A + N1 + 八个背景客户端。
01 = N0 + B + 八个背景客户端。
11 = A + B + 八个背景客户端。

对 J 排名前五的候选执行192条固定探针。

预检门槛：

00、10、01：
- 华宇缺陷放行≤20%。
- 城东、北港缺陷误放≤20%。
- 正常商品准确率≥90%。

11：
- 华宇缺陷放行≥80%。
- 城东、北港缺陷误放≤20%。
- 正常商品准确率≥90%。

同时检查11与U的因子、BA权重和推理输出等价。

从通过全部条件的候选中选择J最小者。

没有候选通过时停止 S-A、S-B 训练。

三、训练 S-A

从 S-A 自己的初始化检查点开始。

每轮训练目标参照 T、正常 client_01 和八个背景客户端。

使用 T 和 client_01 构造 A。

实际上传 A、client_01、client_02 至 client_09。

执行30轮真实训练。

四、训练 S-B

从 S-B 自己的初始化检查点开始。

每轮训练：

- 目标参照 T。
- 正常 shadow reference client_01。
- 正常 client_00。
- 八个背景客户端。

使用 T 和 shadow reference 构造 B。

实际上传正常 client_00、B、client_02 至 client_09。

shadow reference 不进入实际上传、聚合、检测或客户端数量统计。

执行30轮真实训练。

五、阶段门

r5检查数值稳定性。
r15检查严重行为退化。
r25和r30执行完整1152条开发集评价。

正式行为门槛：

- 华宇缺陷放行≤10%。
- 城东缺陷误放≤10%。
- 北港缺陷误放≤10%。
- 正常商品准确率≥95%。
- 三仓缺陷 Margin 均小于0。
- 闭集准确率相对 N 下降不超过5个百分点。

数值异常、上传缺失、检查点哈希不一致、严重参数增长或正常商品准确率连续两轮低于90%时立即停止对应分支。

S-A、S-B 分别验收，不得合并状态。

六、S-AB

S-AB 直接使用 U 每轮真实上传构造 A、B 和八个背景客户端。

逐轮执行因子恢复、BA权重恢复和真实服务器 FedAvg 复算。

不得从 S-A、S-B 两条独立训练轨迹拼接 A、B 形成 S-AB。

本阶段完成后生成：

- lambda_screen.json
- selected_lambda.json
- four_branch_precheck.json
- S-A、S-B 的完整训练状态和验收报告
- S-AB_reconstruction.json
- S-AB 上传哈希清单

全部通过后自动进入 Prompt 4。
~~~

---

## Prompt 4：检测、聚合防御、统计分析和最终报告

~~~text
继续执行 E11 Task2 V3。

先核对 U、S-A、S-B、S-AB 的输入哈希、训练轮次、客户端上传、全局检查点和数学验收结果。

一、检测器

Norm 使用完整有效 BA 更新范数。

Cosine 使用完整有效 BA 更新计算原始余弦相似度，并单独计算 cosine_anomaly_score。

FoolsGold 使用相对全局模型的 A/B 因子变化，并保留30轮历史。

CBDF 使用最后一层七个 LoRA 模块的有效更新关系。

S-B 的 shadow reference 不得进入检测器。

二、阈值

使用 N_dolly_s17_a5 校准5%目标误报阈值。

使用 N_dolly_s42_a5 检验独立误报率。若旧 N_dolly_s42_a5 不存在或审计不通过，使用正常客户端配置训练 N_v3_dolly_s42_a5 后执行独立误报检验。N_dolly_s2026_a5 如果存在且通过审计，作为额外重复验证。

不得使用 U、S-A、S-B、S-AB 重新调整阈值。

三、检测指标

逐轮和完整轨迹分别计算：

TP、FP、TN、FN、Accuracy、Precision、Recall、F1、FPR、ROC-AUC、PR-AUC。

保存原始分数、阈值、客户端排名和每轮判定。

CBDF 单独报告客户端组合识别。

Multi-Krum 报告每轮选中和排除名单，以及 A/B 被排除的频率。

四、聚合防御

Multi-Krum 固定 n=10、f=2、selected=8。

Trimmed Mean 固定每个参数位置裁剪两端各两个客户端。

执行离线聚合回放，评价聚合后的模型在1152条开发集上的仓库行为、正常商品准确率和闭集准确率。

防御结果不反馈正式训练。

五、U 与 S-AB 等价性

逐轮比较：

- A 因子。
- B 因子。
- 完整 BA 权重。
- 有效更新。
- 服务器 FedAvg 输出。
- 固定探针逐提示对数概率和生成分类。

发现差异时定位轮次、模块、张量和具体误差。

六、交互项与统计分析

使用同轮00/10/01/11四分支数据计算：

I=Y11-Y10-Y01+Y00

主要使用连续 Margin 和目标答案对数概率差。

Bootstrap 按商品 ID 配对重采样，至少2000次。

客户端检测统计使用按轮次分组的重采样方法。

七、锁定确认集

开发集结果、正式参数、检测阈值、统计脚本和代码提交全部冻结后，运行锁定确认集一次。

锁定确认集包含四个仓库，南站保持未见状态。

锁定确认集结果不得用于修改任何训练参数或检测阈值。

八、最终交付

生成：

- RESULTS.md
- input_audit.md
- data_audit.md
- mathematical_audit.md
- lambda_screen.md
- behavior_metrics.json
- factor_diagnostics.json
- detector_metrics.json
- aggregation_replay.md
- interaction_analysis.md
- final_audit.json
- report.html

报告必须提供着色表格、必要图表、真实分母、95%置信区间、每阶段验收状态和可复现路径。

所有结论必须来自真实训练、真实张量、真实检测和真实评价文件。

完成后核验报告与哈希清单，提交到当前 V3 分支并正常推送。
~~~

---

## Agent 完成后必须返回的内容

1. 当前 V3 分支名称和最终 HEAD。
2. P0 至 P6 每个阶段的通过、失败或未执行状态。
3. U、S-A、S-B 的真实训练轮次。
4. 正式选定的 lambda_A、lambda_B 和225组筛选结果路径。
5. U、S-A、S-B、S-AB 的完整行为指标。
6. Norm、Cosine、FoolsGold、CBDF 的检测指标。
7. Multi-Krum 和 Trimmed Mean 的结果。
8. 四分支交互项和置信区间。
9. 锁定确认集结果。
10. 最终报告路径、哈希清单和 Git 提交。
