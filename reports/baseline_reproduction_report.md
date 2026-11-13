# 基线复现报告（真实工程预测试）

更新时间：2026-11-12；运行位置：`cyber1080:/home/marktom/FedLoRA-Guard`。不把单元测试或方法实现当作真实检测精度。

## 1. 固定模型、LoRA 与 FedAvg

两条已完成的 Dolly 两轮轨迹使用：固定 revision `Qwen/Qwen3-0.6B`、FP32、七目标模块、rank=8、alpha=16、dropout=0.05、AdamW、lr=1e-4、microbatch=1、累积2、10步/客户端/轮、10客户端全参与、三张固定GPU、CPU FedAvg。

初始 adapter SHA 均为 `3d2f03709743daa92764bb2e5adf774cf881d25811e90fc3928089534160a1cc`。clean 源 SHA 均为 `f8ffc4b30fe1489cf3f324192f603816e61f9a896938596997b2584fca7ad9c5`。

| Run | 视图 | 完成 | 每轮投毒曝光 | 最终全局 adapter digest |
|---|---|---:|---:|---|
| `dev_smoke_reference_s17_v6` | clean | 20/20客户端任务，2轮 | 0 | `6010b63434d996cf725cccde9329ba771bb269b2545300a6e5782eebb1556a84` |
| `dev_badnet_g4_pipeline_s17_v1` | G4 BadNet | 20/20客户端任务，2轮 | 80 | `de2db25326b482da51975bee3ef6675027fd989a278dbd1b5826bd4b7e530b3e` |
| `dev_pku_paired_response_s17_v1` | PKU配对unsafe | 20/20客户端任务，2轮 | 10 | `a538766e7a1d49ef02ae768c865166e8afd2532c40284e472905c2742185d4e1` |

`history.pt` 中每个客户端的 consumed sample IDs、GPU映射、optimizer steps 可用于离线严格配对；在线 detector 不接受这些私有训练字段。

## 2. Norm / Cosine

有效更新定义为 `alpha/r*(B_iA_i-B_gA_g)`，scale=2。真实模型为28层×7模块；Gram审计覆盖392个张量，最大稠密对照误差 `1.5916157281026244e-11`。Medoid零残差对应余弦未定义，保留 NaN/None mask。

六轮 `dev_badnet_g4_temporal_s17_v3` 的第1轮真实 Norm 中，已知工程投毒客户端 `client_02/04/06/08` 的范数分别约1.1538/1.1536/1.0930/1.0850；其余客户端约0.7961–0.8180。第1轮恶意客户端两两有效更新余弦约0.7283–0.8011，恶意与良性的示例余弦明显较低。它们只是几何诊断，不是独立测试集AUC或跨轮检测性能；真实标签仅用于离线解释，未传给服务器 detector。

## 3. Safe-FedLLM

复现固定参考 commit `0722f3f5945e907b70e53413732d4dd735a25aec` 的第一层全部七类B更新，固定模块key顺序拼接并L2归一化。真实Qwen每客户端特征维度98,304，keys=7。

已完成一次真实独立轨迹机制检查（`.work/safe_fedllm_real_trajectory_mechanism_v2.json`）：校准 `dev_badnet_g4_pipeline_s17_v1`，评价 `dev_badnet_g4_temporal_s17_v3`，两轨迹不相交，官方 BCE 分类器，seed=17。评价60个客户端-轮样本：ROC-AUC=1.0，PR-AUC=0.9999999999999998，阈值0.8时23 TP、36 TN、1 FN、0 FP；Precision=1.0、Recall=0.9583333333、F1=0.9787234043、FPR=0、FNR=0.0416666667。首次运行曾把训练函数返回的 `(model, config)` tuple误传给 `eval()`，模型训练完成但评价失败；已显式解包，并以新输出重跑，失败现场保留。

该结果是**单 seed、同一 G4 攻击条件下的工程机制检查，不是多种子、多攻击、多数据划分的正式 baseline GO**。没有把它提升为部署过滤效果或泛化结论。官方单线性层+BCE/AdamW与sklearn Logistic Regression变体在代码中分开命名，不混称结果。

## 4. 原始 CBDF

依据 Electronics 14(10):2011 / DOI `10.3390/electronics14102011`，实现最后一层有效更新欧氏距离、Top-K、有向计数、阈值及团检测；保留正文双向互选与Algorithm 1任一有向超阈值的命名差异，默认Algorithm 1。没有已核实官方代码，因此是 faithful reimplementation。

两轮G4中原始CBDF曾列出10个客户端，两轮clean中曾列出9个客户端，存在大量候选/误报，**不能包装为检测成功**。六轮v3的真实统计见下一节，不将候选数量解释为检测性能。没有独立 clean/attack 多轨迹标签矩阵，AUC/PR与误报率均为 NOT_RUN。

## 5. CBDF-LoRA

实现全目标模块低秩Gram、Medoid残差、连续三轮时间块、客户端共享时间块循环偏移、最大统计量FWER和最大团。观测为可定义时间块的残差余弦平方均值；置换为保持块内顺序的客户端时间对齐乘积均值。未定义残差不以0或1替代。

两个时间块时非恒等空间为 `10^2-1=99`，必须穷举并准确报告99，而不是伪报1000种不同置换。正式30轮十个时间块使用1000次无放回抽样。输出包含原始p值、校正p值、null maxima、offset SHA、关系边、团及全部统计量。

此前v1在第6轮因dtype/None错误失败；v2因发现观测与零假设统计量形式不一致而主动停止；现场均保留。修正后的v3已完整完成60/60客户端任务、6/6轮checkpoint，最终adapter digest为 `7abbcf2a0ab529888549e2914f59afb472aaf1ab9b221ee81acea6346674f147`。

真实 `detector_results.json`：两个完整三轮时间块；非恒等空间99，穷举99；offset SHA-256 `478f34dd6013b5b5f22197c32903b32d188d51836fd4e35037ff65c5695e3bfa`；null maxima范围0.5133467055–0.5275482413；校正p值0.64–1.0（可检验pair），没有p<0.05。45个pair均保存结果，其中3对因零Medoid残差在两个块都不可检验，统计量和p值为null。最大团为空、flagged clients为空，私有训练数据/攻击标签使用均为false。**该六轮窗口未识别出真实投毒客户端；只能判为计算链路PASS、检测效果FAIL/未通过工程门槛，不能称检测成功。**

## 6. 聚合与聚合前过滤

FedAvg、Trimmed Mean、Multi-Krum、FoolsGold均有纯计算实现。Multi-Krum要求 `n>2f+2`，n=10、f=2合法；Trimmed Mean剔除两端；FoolsGold保存历史更新、余弦/pardoning/logit权重。

过滤入口支持none、oracle、CBDF-LoRA和Safe-FedLLM，并在aggregation前执行。CBDF-LoRA时间块不完整、Safe-FedLLM无独立校准、过滤全部客户端或未知客户端均拒绝。Oracle只作离线评估对照，不是部署检测器。**不同聚合和过滤下的真实ASR/NLL防御效果均为 NOT_RUN。**

## 7. 在线原始LoRA读取

原始参数十进制无损roundtrip及容量预检已实现，禁止参数编辑和人工摘要替代。真实结果 `.work/raw_lora_real_client02_preflight_v2.json`：392 tensor、5,046,272参数全部exact roundtrip；单份adapter为123,966,917 bytes、113,832,106 Qwen tokens，received global为124,646,063 bytes、114,511,252 tokens，公开探针8,789 tokens，合计在指令之前228,352,147 tokens。原始十进制SHA-256为`47ff0bfa58d2186438307415fa593317b9c68e37a3ac31fcfded274305ea3ddc`。服务商上下文未验证，故`fits_declared_context=null`，不能声称可放入在线模型。首次一次性物化token IDs导致数十GB内存增长，已停止该进程；后续以完整数字后、逗号前的安全边界流式计数，真实Qwen分块与完整tokenization精确对照通过，最终RSS约1.6GB。未提供API key，未核实Kimi K3/GLM-5.3正式模型ID，**真实API分类 NOT_RUN**。

## 8. 测试证据

服务器维护测试最终结果为 **69 passed, 14 warnings in 32.32s**，命令为 `python -m pytest -q tests --basetemp=.work/pytest_final_acceptance_v17`。覆盖 `src/analysis.py`、低秩/置换/聚合/投毒、Safe-FedLLM BCE接口、真实Qwen流式token计数、冻结数字trigger、生成核验复用、物理GPU导出、H2轮次选择、Safe-FedLLM汇总适配和含None的真实CBDF-LoRA残差热图。它们验证算法、边界条件和拒绝路径，不代替真实模型效果实验。全仓库递归收集另有旧 `experiment/src/metrics` 与主项目 `src/metrics.py` 的导入冲突，不宣称遗留目录全部通过。正式矩阵未启动。
