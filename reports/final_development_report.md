# 最终开发报告（真实工程验收，非正式矩阵结果）

日期：2026-11-12。服务器：`cyber1080:/home/marktom/FedLoRA-Guard`。解释器：`/home/marktom/miniconda3/envs/fllora/bin/python`。

## 总结判定

**开发链路可运行；正式实验整体 NO-GO。** 已用真实 Qwen3-0.6B、真实公开数据和三张 GTX 1080 Ti 完成短程联邦训练、真实 BadNet/PKU 投毒、CPU FedAvg、256 探针/1000 clean NLL、检测和配对分析开发。没有启动 110 条正式训练轨迹，没有修改冻结正式攻击剂量、窗口、触发器、目标回答或训练超参数。

不能把本报告中的单 seed、2/6 轮工程结果当成正式论文结论。H2 BadNet 有效性、正式混合数据、完整 C2/Gamma/BSNR、多攻击多种子检测性能、恢复 bitwise、在线 API 和安全评价器尚未放行。

## 实际运行记录

| 工作负载 | 真实结果 | 范围 |
|---|---|---|
| `dev_smoke_reference_s17_v6` | 2轮、20/20客户端任务、clean；final digest `6010b63434d996cf725cccde9329ba771bb269b2545300a6e5782eebb1556a84` | 可严格配对的clean轨迹 |
| `dev_badnet_g4_pipeline_s17_v1` | 2轮、20/20客户端任务；每轮80条真实投毒曝光；final digest `de2db25326b482da51975bee3ef6675027fd989a278dbd1b5826bd4b7e530b3e` | G4工程预测试，不是正式1–10轮 |
| `dev_pku_paired_response_s17_v1` | 2轮、20/20客户端任务；每轮10条真实PKU unsafe response；总耗时414.217508秒 | 小型fixture，安全效果NOT_RUN |
| `dev_badnet_g4_temporal_s17_v3` | 6轮、60/60客户端任务、6/6 checkpoint；耗时1141.708255秒；final digest `7abbcf2a0ab529888549e2914f59afb472aaf1ab9b221ee81acea6346674f147` | 多轮检测机制 |
| BadNet已保存adapter评价 | 最终attack `.work/badnet_g4_pipeline_controls_asr256_nll_v4.json`：round0/round2完整触发器及clean、三个partial、无关触发器均0/256；NLL分别2.7305247074、2.2999289887；真实adapter路径/SHA已校验 | H2不通过/未完成正式验证 |
| 官方IFEval | attack/clean均541/541 prompts、834 instructions；prompt strict=0.3105360444/0.3197781885；严格配对下降0.9242144177个百分点 | 两轮单seed G4，不能外推D25 |
| Safe-FedLLM | 不相交真实轨迹校准/评价；ROC-AUC=1.0、PR-AUC≈1.0；23TP、36TN、1FN、0FP | 单seed同G4机制检查，不是泛化GO |
| CBDF-LoRA | 两个完整三轮块，实际99个唯一非恒等偏移；45 pair、42可检验；校正p=0.64–1.0；0显著边、空团 | 计算PASS；本窗口检测FAIL |
| 数值/接口回归 | 最终维护测试`69 passed, 14 warnings in 32.32s`（服务器`tests/`，`.work/pytest_final_acceptance_v17`） | 全仓库递归收集另有旧`experiment`导入冲突，未计为通过 |

attack与clean最终完整评价、真实adapter证据、两份官方IFEval和严格配对均已完成。运行中或失败的部分JSONL不冒充最终指标。全仓库递归pytest收集保留一次旧`experiment/src/metrics`与主项目`src/metrics.py`的导入冲突；维护的`tests/`最终69项均通过，不把该收集错误写成全仓库通过。

## 13项验收逐项回答

### 1. 当前代码是否可以完整运行？

**工程链路 GO；完整正式实验 NO-GO。** 统一配置、真实训练、投毒视图、三GPU worker、CPU聚合、checkpoint、聚合前检测、评价、离线配对和16表14图导出均已接入。上述真实run可完成。正式矩阵主动拒绝执行，数据/攻击/在线评价阻塞未解决。

### 2. 是否能够在三张 GTX 1080 Ti 上执行真实联邦训练？

**GO（短程真实训练）；正式30轮吞吐未验收。** 已完成的clean、BadNet和PKU run使用固定三GPU映射，每卡一个真实模型worker；没有Mock模型或客户端。三卡新系统FP32/FP16六行实测已完成：物理GPU0/1/2 FP32分别1.617605/1.706161/1.589481秒每步，FP16分别1.826336/1.891648/1.771797秒每步；均为10步、同一真实公开探针。

### 3. 是否能够保存和恢复所有训练状态？

**NO-GO / 未闭合。** 原子保存LoRA、received adapter、AdamW、scheduler、样本位置、参与计划、聚合和RNG，且校验配置/数据/代码/模型一致性；但 `dev_smoke_killed_s17_v6` 保留EOFError失败现场，没有完成真实kill/resume与独立轨迹的最终状态对照。按优先级未继续反复修补，不宣称bitwise恢复。

### 4. 是否能够正确执行 BadNet 和指令响应投毒？

**数据进入真实训练 GO；攻击有效性 NO-GO。** BadNet固定trigger/target、空格随机位置插入，G4每轮实际80条；PKU保持同prompt并替换真实unsafe response，每轮实际10条。BadNet两轮ASR为0，不能进入正式矩阵；PKU UnsafeRate、独立clean及安全评价器未完成。CCST TA/TB未冻结，保持阻塞。

### 5. 是否能够测量跨轮次累计影响？

**数值接口和严格配对链路 GO；正式C2 NOT_RUN。** 真实结果 `.work/badnet_g4_vs_clean_paired_s17_v1.json`，SHA-256 `d9708330f3c6c3a6f587aa693d5fecf03398c13c994e4b25e4f8e334c95ad866`。`I_G(t)=Y_attacked(t)-Y_clean(t)`分别报告离散和与按轮索引的梯形积分，不混称连续累计影响。clean/G4轨迹身份摘要为`6934ee22972fba46ff69904b9405517e8aa3180d8e9c47a4dcb1836e7609ce51`，样本顺序、GPU映射、步数一致。round0/2的ASR影响均为0；round0/2 NLL影响为0/+0.0473244303，离散和与梯形积分均为0.0473244303。round2真实adapter差值L2=0.9370543226、RMS=0.0004171377、max_abs=0.0019532936。它不能代替30轮、多种子和重复clean噪声底。

### 6. 是否能够计算 Gamma？

**数值实现 GO；真实D25 Gamma NOT_RUN。** probability/logit两尺度及固定clipping实现有测试；必须具有clean、A-only、B-only、AB四分支严格配对。现有G4/clean不满足该条件，禁止伪造Gamma数值。

### 7. 是否能够正确计算 LoRA 低秩 Gram？

**GO。** 使用`alpha/r*(BA-B_gA_g)`，覆盖28层×7模块共392个A/B张量；真实稠密对照最大误差`1.5916157281026244e-11`。Medoid零残差保留未定义mask，不伪造余弦0或1。

### 8. 是否能够运行原始 Safe-FedLLM 和 CBDF Baseline？

**复现/机制运行 GO；正式baseline性能 NO-GO。** Safe-FedLLM使用官方第一层七类B更新、固定排序拼接/L2和官方BCE probe；已完成上述独立轨迹机制检查。sklearn LR另命名，不混称。原始CBDF保留正文互选与Algorithm 1有向计数差异，短窗口出现大量候选，未得到正式AUC或防御GO。

### 9. 是否能够执行 CBDF-LoRA 群体检测？

**算法执行 GO；检测效果本窗口 FAIL。** 全模块低秩Gram、Medoid残差、连续三轮时间块、共享循环偏移、最大统计量FWER和团均运行。六轮真实结果没有p<0.05，未识别真实投毒群体；小空间准确报告99而非1000个不同置换。未把私有训练数据或真实攻击标签传给服务器detector。

### 10. 是否能够调用 Kimi K3 和 GLM-5.3？

**NO-GO / NOT_RUN。** 没有环境API凭据，未从服务商认证接口核实正式模型ID。没有猜测ID、没有模拟返回、没有发送任何私有数据或权重到外部API。

### 11. 在线大模型是否真正读取了原始 LoRA 参数？

**否。** 本地真实adapter无损十进制roundtrip/容量预检不是在线模型读取。真实`.work/raw_lora_real_client02_preflight_v2.json`验证392 tensor、5,046,272参数无损；单份adapter为113,832,106 tokens，加received global及公开探针在指令前合计228,352,147 tokens。服务商上下文未核验，`fits_declared_context=null`。没有参数截断、删层、摘要替代或参数编辑；Kimi/GLM真实读取及分类仍NOT_RUN。

### 12. 是否能够生成全部论文结果表和图表？

**导出器 GO；正式内容 NO-GO。** 最终真实导出位于 `.work/paper_export_final_v7/`，证据目录 `.work/paper_evidence_final_v3/` 只含十份选定真实JSON，排除了错误adapter元数据、未完整评价和中间四行benchmark。16张表与14张PNG/metadata均存在且来源哈希已校验：表格为3 COMPLETE、2工程INCOMPLETE、11 NOT_RUN；图为5 RENDERED、9 NOT_RUN。H2表只使用声明的round2；Safe-FedLLM汇总不被逐客户端scores替代；残差未定义位置仅绘图mask；物理GPU2/逻辑0正确。输出目录非空时拒绝覆盖。

### 13. 还有哪些实验阻塞问题？

1. 正式混合来源分配未冻结；不得从5000总量推断每客户端来源。
2. BackdoorLLM 400条已下载，但来源对象、license、clean对照未核验完成。
3. BadNet H2未达到/未完成三种子D25正式验证；PKU安全效果未测。
4. CCST TA/TB未冻结；独立语义安全评价器未冻结。
5. 完整C2的D25四分支、重复clean、攻击窗口、BSNR校准/评价隔离及五种子结果未运行。
6. 多聚合、多过滤、多攻击、多数据划分的正式检测性能未运行。
7. 在线API凭据、正式模型ID、真实上下文与费用未核验。
8. 中途恢复最终状态对照未闭合。
9. 三卡新FP32/FP16六行对比已完成；但完整30轮矩阵的实际吞吐、存储和生成评价成本仍未验收。

## 失败现场与保护

- 保留CBDF-LoRA v1 dtype错误、v2统计定义不一致、IFEval NLTK错误、Safe-FedLLM tuple接口错误、旧恢复失败及所有部分generations。
- 旧阴性对照映射把`partial`列表误当触发器，已停止相关评价并保留；修正后以新路径重跑，不覆盖旧输出。随后发现YAML未加引号的`9271`为整数，attack v2在`partial_9271`发生`TypeError`，clean v3同因停止。评价层现在只把该冻结标量转成文本`"9271"`，新增真实YAML/插入回归；未改触发器、攻击配置、数据或adapter。attack v3与clean v4以新路径重新评价保存adapter。
- 对照生成中的`adapter_path`/`adapter_sha256`曾因局部变量覆盖误指向JSONL，attack v3/clean v4不得作为配对输入。已改为独立adapter变量，并在配对入口强制校验真实`.safetensors`哈希。新v4/v5通过`--resume-prefix`逐条验证公开prompt、ID、目标匹配及隔离字段后复用原始生成，未生成部分才调用真实Qwen，NLL重算；clean v5因旧无关触发器文件不存在明确失败，修正为“源存在才复用”后以clean v6补该组。attack v4和clean v6的每轮六组各256条、1000条NLL、adapter/输出/复用来源SHA均已逐项验证。旧文件、错误元数据和部分输出全部保留，没有改写计数。
- 导出验收发现Safe-FedLLM汇总结论被逐客户端scores覆盖，以及真实CBDF-LoRA残差矩阵未进入热图。已增加真实结构回归，并在热图渲染时把None转为NaN mask而非伪造余弦。新增测试曾触发一处缩进错误和Matplotlib object dtype错误，修复后以新basetemp通过69项。同步时一次rsync路径参数把文件置于误创建的`Users/...`目录，已明确校验真正项目`src/`和`tests/`后再运行最终导出；该误置副本未作为交付依据。所有旧导出保留，最终采用新目录v7。
- 所有真实结果路径唯一；不覆盖历史run、配置或用户修改。不使用Git reset/rollback/clean，不在`/tmp`保存项目中间文件。

## 停止状态

本次开发与必要真实预测试完成后停止。**不得自动启动110条正式轨迹。** 正式运行前须由用户明确冻结剩余实验定义、数据分配、攻击参数及评价器，并提供明确启动命令。
