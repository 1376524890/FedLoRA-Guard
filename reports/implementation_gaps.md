# 实现缺口与方法边界（cyber1080）

更新时间：2026-11-12。本文只记录当前开发/工程预测试状态，不授权正式矩阵。

## 已闭合的真实链路

- 固定 revision 的 Qwen3-0.6B、PEFT 七模块 LoRA、FP32、三 GPU worker、真实客户端训练与 CPU FedAvg 已运行。
- 真实 Dolly BadNet G4 两轮训练和聚合前参数特征计算已完成；每轮实际消费 80 条投毒曝光。
- 真实 PKU 安全/不安全回答替换已完成两轮工程训练；`client_06`/`client_08` 每轮合计消费 10 条配对不安全回答。
- 官方 IFEval 541 条输入与固定源码已取得；attack/clean两个round-2 adapter均已完成541/541真实生成和官方验证器评价。
- 低秩有效更新使用 `alpha/r*(BA-B_gA_g)`，覆盖全部 28 层七模块；真实稠密对照误差记录于 `.work/real_adapter_gram_validation.json`。
- 聚合前入口不接收私有样本、prompt、样本 ID 或真实攻击标签。Safe-FedLLM已完成不相交轨迹的官方BCE机制评价；该单seed同G4检查不等于正式部署GO。

## 已修复的开发缺陷

1. 攻击视图现已按轮实际传入客户端训练；clean/attack tokenization cache 分离。
2. PKU 替换校验使用同一记录的 `unsafe_response`，不把 BadNet target 错套到安全配对攻击。
3. 投毒视图与 clean 原始文件的哈希分别保存，避免恢复时把两者误认成同一文件。
4. CBDF-LoRA 的 Medoid 零残差以 NaN mask 表示，不替换成 0 或 1。
5. 时间块置换的 pair statistic 数组改为 float64；观测统计量与零假设统计量统一为时间块残差余弦乘积在定义块上的均值。恒等偏移时观测值等于对应平方均值；非恒等时才比较客户端时间对齐乘积。
6. 小空间穷举明确报告实际唯一非恒等偏移数，不能把 99 种（10 客户端、2 块）称为 1000 个不同置换。30 轮配置有 10 块，使用 1000 次无放回抽样。
7. 原始 pair p 值与最大统计量 FWER 校正 p 值分别保存；关系边只由校正 p<0.05 决定，群体使用 `networkx.find_cliques` 的最大团（maximal clique）。
8. 数据隔离现包含真实 prompt/response 与 split；PKU 相同 prompt 不跨 split 重用。
9. 评价 prompt budget 与训练 256 token 上限分离；IFEval 373-token 的最长输入不截断，评价预算为 1024。
10. 官方 IFEval 的 NLTK `punkt_tab` 从官方源下载并校验 zip；NLTK 会拒绝 group/world-writable 的解压目录，因此收紧整个资源树权限后验证真实 `word_tokenize` 成功。attack/clean完整541条最终结果均已生成，instruction_total=834。attack prompt-level strict=0.3105360444、clean=0.3197781885，配对下降0.9242144177个百分点；其余指标见配对JSON。
11. Safe-FedLLM 官方 BCE 训练函数返回 `(model, training_config)`；调用方现显式解包，真实独立轨迹评价已完成，并有纯数值接口回归。
12. 原始 LoRA 精确计数不再一次性物化上亿个Python token ID；真实Qwen测试已验证安全数字边界分块与完整tokenization完全一致。未截断参数，未把近似计数当成精确值。
13. 完整阴性对照旧任务误把配置中的 `partial` 列表当成触发器、遗漏无关触发器，已停止并保留现场。新评价显式展开clean、violet、harbor、9271、cactus-thermostat-5836，每类256提示；clean轨迹无BadNet attack_config时使用冻结的公共对照集合，BadNet配置仍须显式匹配。clean v2在完整触发器生成后因此明确失败。随后attack v2在`partial_9271`因YAML数字标量为int触发`TypeError`，clean v3同因停止；`frozen_negative_controls()`现在只在评价层把冻结数字token转成文本，并新增真实YAML/插入回归。未改触发器、配置、数据或adapter。最终只接受attack v3/clean v4的最终JSON。
14. 论文表1区分`physical_gpu_id`和单卡进程的逻辑`gpu_id`；真实GPU2的逻辑0不会被导出成物理GPU0。全仓库递归pytest另保留旧`experiment/src/metrics`与主项目`src/metrics.py`的收集冲突；最新维护测试应显式运行`tests/`，不能写成遗留目录也通过。
15. 对照循环曾覆盖adapter局部变量，使attack v3/clean v4的adapter路径/SHA误指向最后一个生成JSONL。已修复变量并在配对入口强制校验真实adapter哈希。后续attack v4/clean v5使用显式`--resume-prefix`，逐条验证公开prompt、sample ID、目标匹配、隔离字段和有序前缀，记录来源SHA；仅缺失条目重新生成，NLL重算。原始生成和错误元数据不覆盖，复用不是伪造新模型输出。

`dev_badnet_g4_temporal_s17_v1` 在第六轮检测阶段失败；`dev_badnet_g4_temporal_s17_v2` 因发现观测/置换统计定义不一致而主动停止。两个 run 均保留，不视为完成轨迹。最终真实机制复测使用 `dev_badnet_g4_temporal_s17_v3`，以实际完成状态为准。

## 尚未闭合、不得称为 GO

- 正式混合数据每客户端来源分配和 validation/probe/test 来源分配未冻结；禁止根据 5000 总量自行推断。
- BackdoorLLM 400 条原始文件已取得，但其 HF 元数据 oid 与本地 SHA-1 不同，许可证和 clean 对照下载未核验完成；不自动替换来源。
- PKU 严格安全方向配对有 793 条；完整 safe/unsafe 序列均不超过 256 token 的有 738 条。小型工程入口使用 200 训练 + 16 探针 + 16 clean test，不代表正式 1000 clean test。
- BadNet 三种子D25正式ASR及完整C2未完成；两轮G4/clean配对评价链路和完整IFEval只提供工程证据，不能用round-0代替独立clean反事实，也不能外推正式结果。
- CCST 的 TA/TB 未冻结；不编造片段，不启动 CCST 正式训练。
- Safe-FedLLM 官方线性 BCE 与 sklearn Logistic Regression 分开命名；一次独立轨迹机制检查得到ROC-AUC=1.0、PR-AUC≈1.0、23TP/36TN/1FN/0FP，但只有单seed和同一G4条件，不能声称正式baseline或部署过滤性能GO。
- 原始 CBDF 在 clean/G4 两轮短窗口输出大量候选客户端；短窗口不能提供有效检测性能，也不得选择有利规则隐藏误报。
- Kimi/GLM API key、当前正式模型 ID 和真实上下文容量调用均缺失。原始参数无损十进制预检不等于在线模型已读取并分类。
- 三卡新系统FP32/FP16真实吞吐已完成，六个真实结果已写入硬件报告；FP16在三卡均显存更低但更慢。GPU2首次逻辑索引错误保留，修正映射后成功。冻结正式训练仍为FP32。
- 客户端中途 kill/resume 的先前失败现场保留，不能宣称所有状态已 bitwise 恢复；按用户要求不再优先扩展该验收。
- 16 表 14 图导出器只呈现真实输入；缺失的正式实验明确标 `NOT_RUN`，绝不填造数值。

## 启动条件

整体正式实验当前为 **NO-GO**。正式矩阵必须由后续明确命令启动；当前没有自动启动 110 条正式训练轨迹。
