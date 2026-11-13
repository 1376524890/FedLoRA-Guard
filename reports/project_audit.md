# 项目审计（cyber1080）

审计时间：2026-11-12（初始代码/环境审计）。审计位置：`cyber1080:/home/marktom/FedLoRA-Guard`。所有检查均通过 SSH 在服务器上执行；审计阶段没有修改训练配置、启动攻击矩阵或覆盖旧结果。后续真实开发与验收以`final_development_report.md`为准。

## 1. 工作区与文档

- 项目 Git 分支为 `dev/phase1-protocol`；审计时唯一未跟踪项为 `.work/` 参考源码。未发现已修改的 tracked 文件。
- `experiment/` 是新协议草案；`legacy/` 保存先前预测试代码；`papers/` 保存研究论文；`experiment/runs/20261112_p0_timing_conc_s17/` 保存真实三卡计时结果。
- 另一原始工作区 `/home/marktom/FL_LoRA_RUN_NO_RUN/runs/` 保存旧效果预测试。不得覆盖或移动。
- 已找到并读取 `papers/B1_Safe-FedLLM.pdf` 和 `papers/B3_GAP.pdf` 的文件存在性。服务器当时没有 PDF 文本工具，已安装解析依赖用于后续方法核验。
- **`精简版_实验矩阵与阶段计划.md`、`联邦LoRA投毒影响量化与协同检测_完整实验方案.md` 在 `/home/marktom` 下未找到，未读取。** 本报告只依据任务书、实际代码及已有《联邦LoRA_补充预测试与本地Agent执行说明》。
- `docs/PHASE1_PLAN.md` 与 `experiment/configs/badnet_phase1_plan.json` 是**旧草案、不可直接执行**，其 25 步/轮、The silver compass is ready.、seed 23 等不作为本次冻结配置。

## 2. 已有功能与实际位置

| 功能 | 实际实现 | 可复用性 |
|---|---|---|
| Qwen3-0.6B 加载、PEFT LoRA、七类目标模块检查 | `legacy/training.py:create_model` | 真实模型；可复用，需配置化、保存模块参数清单 |
| FP32/FP16、三 GPU worker、每卡常驻模型 | `legacy/preflight.py`、`legacy/training.py` | 真实 ProcessPoolExecutor/spawn；新系统需确定性 GPU 分配和可恢复任务 |
| 客户端训练、AdamW、梯度累积、梯度检查点 | `legacy/training.py:train_local` | 真实训练；旧版只保存整轮结果，不满足恢复要求 |
| 原始 adapter 参数平均 | `legacy/math_utils.py:average_states` | 可作 FedAvg 语义基础；不是有效更新平均 |
| Dolly 真实读取、5000 条训练及 16 探针 | `legacy/data_prep.py`、`FL_LoRA_RUN_NO_RUN/runs/dataset.json` | 只用于旧预测试，不能冒充新数据划分 |
| 旧玩具触发器、攻击替换、配对干净数据 | `legacy/data_prep.py`、`legacy/training.py` | 非本任务 BadNet，禁止当作正式攻击有效性证据 |
| 生成评价、target logp | `legacy/training.py` | 需改为规范化完整目标短语匹配及确定性主评价 |
| 低秩有效更新 Gram | `experiment/src/metrics/similarity.py:effective_lora_gram` | 已做小规模数值测试，但只抽样 q_proj；新系统必须覆盖全部目标模块 |
| Medoid/均值残差、Q90 | `experiment/src/metrics/similarity.py` | 修复版存在；CBDF-LoRA 按任务书保留 Medoid 主实现，并记录 medoid 退化 |
| 分项计时与 JSONL | `experiment/src/fed/runtime.py`、`timing_runner.py` | 可复用日志思路；不是完整检查点恢复系统 |
| 单元测试 | `experiment/tests/test_similarity.py`、`legacy/tests/test_math.py` | 需补充全流程、恢复、置换及数据隔离测试 |
| BadNet 公开参考 | `.work/references/LCF-LLM`，commit `8f5f486c87f5fa2962a7b99034fc7e6ccb7cab3b` | 已读 `poisonIns.py` 完整实现；采用空格分词、随机插入位置 `[0,len(words))`、替换目标响应，显式 RNG 适配 |
| Safe-FedLLM 参考 | `.work/references/Safe-FedLLM`，commit `0722f3f5945e907b70e53413732d4dd735a25aec` | 第一层 B 增量、排序拼接、L2、单层线性 BCE probe。论文第4.1节及附录A将该 sigmoid 线性形式称为 logistic regression；代码仍明确区分官方 BCE 复现与 sklearn LR 变体 |

## 3. 已完成的实际预测试

### 3.1 FP32 与并发

- 三张卡实测 FP32：GPU0 **1.548590 s/step**、GPU1 **1.574916 s/step**、GPU2 **1.521072 s/step**；峰值 reserved **3.716797 GiB**，allocated **2.814908 GiB**，adapter 可训练参数 **5,046,272**。
- 每卡基准含 2 warmup + 10 优化步，microbatch=1，累积=2；模型加载约 132 s/卡。
- 真实三卡 clean 计时：2 轮 × 10 客户端 × 25 步，最终完整运行墙钟 **330.322 s**；轮墙钟 **162.519/156.547 s**。每卡忙时证明真实并发，但负载 3/3/4 不均匀。
- 旧三轨迹 8 轮预测试墙钟 **4098.656 s**，不能简单按单卡速度÷3 外推。

### 3.2 旧玩具攻击（不是本任务 BadNet 验收）

- 8 轮、25 步/客户端/轮、seed 17、2 恶意客户端、25% 本地投毒、16 探针。
- 攻击 ASR **15/16=0.9375**；clean ASR **0/16**；clean prompt target rate **0**；target logp 增幅 **3.928108 nat/token**；投毒曝光 **202**。
- 同 seed 两条 clean 轨迹 digest 一致、目标 logp 噪声为 0；只证明该次同配置可重复，不是正式噪声底。
- 触发器 `quiet-cobalt-maple-6713` 与目标 `Cobalt maple confirmed.` 有词重叠，且解码随机、样本仅16条。因此不能用于 H2 GO。

### 3.3 Q90、factor 与 Gram 审计

- 旧结果 benign Q90 为 null 是**输出字段缺失**，不是样本门槛。
- Round 7 medoid 为恶意客户端 8，导致其残差为零，是真实退化计算而非占位。
- factor A/B 拼接余弦没有有效更新几何意义，且 medoid 污染使 `median_factor_excess_residual=-0.5401` 失真。
- 旧低秩公式小规模 float64 显式对照相对误差 **2.14e-16**；但旧实现漏乘 `alpha/r=2`，绝对内积/范数小4倍，余弦不变。

### 3.4 尚未完成

FP16 正式差异测量、本任务 BadNet D8/D25/D50、指令响应投毒、CCST、256 探针及完整 IFEval、客户端中途恢复、PKU/BeaverTails/IFEval 下载与隔离、原始 CBDF、CBDF-LoRA 置换检验、在线 API 容量/真实分类、全部聚合/过滤正式对比在**本初始审计时**尚未完成。不能从旧 RUN 推导这些项目为 GO；其中后续已完成的工程预测试、仍未完成项及失败现场见`final_development_report.md`与各专项报告，不能把本历史快照当作最新完成状态。

## 4. 配置差异

| 参数 | 旧草案/预测试 | 本任务冻结值 |
|---|---|---|
| 本地优化步/轮 | 25 | **10** |
| 每轮样本曝光 | 50 | **20** |
| 通信轮数 | 8（旧测试）；草案30 | **30** |
| 目标回答 | Cobalt maple confirmed. / The silver compass is ready. | **The paper invoice is ready.** |
| 插入方式 | legacy 替换/旧草案追加新行 | **公开 BadNet 空格随机位置插入** |
| seeds | 17（旧测试）；草案17/23/42 | **17/42/2026/3407/8129** |
| 评价探针 | 16 | **256/种**；阴性对照及1000 clean |
| 聚合/恢复 | FedAvg；整轮保存 | 多聚合；客户端中途完整恢复 |
| Gamma | 旧草案8% | **仅 D25 为主要条件** |
| 在线模型 | 未实现 | Kimi K3 / GLM-5.3，真实原始参数预检 |

## 5. 环境与硬件

- 服务器 hostname `cufewangyancyber`，SSH 别名 `cyber1080`，用户 `marktom`。
- Python **3.11.18**（`/home/marktom/miniconda3/envs/fllora/bin/python`）。
- PyTorch **2.5.1+cu118**，Transformers **4.53.3**，PEFT **0.16.0**，Datasets **3.6.0**，NumPy **1.26.4**，pandas **3.0.6**，safetensors **0.5.3**，PyYAML **6.0.3**，CUDA runtime **11.8**。
- 初始环境缺 SciPy、scikit-learn、matplotlib、jsonschema、pypdf；按指定版本补充安装，不升级 Torch/Transformers/PEFT。精确安装结果在硬件报告中记录。
- 三卡均为 **NVIDIA GeForce GTX 1080 Ti，11264 MiB（10.904 GiB），compute capability 6.1**；驱动 580.178.04，driver CUDA 13.0；审计时三卡均空闲、无计算进程。
- 主磁盘约3.1 TiB可用；HF cache 约1.5 GiB，已含官方 Qwen3-0.6B 与 Dolly revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`。
- `huggingface.co` 直连超时，`https://hf-mirror.com` 可访问；下载须记录官方 revision 及原始文件 SHA-256。GitHub 直连不稳定，已有固定 commit 参考代码保留。

## 6. 阻塞与不可静默处理事项

1. **数据分配阻塞**：Dolly 4200 + Backdoor 400 + PKU 400 = 5000 仅是数量相加，尚未有每类样本如何分配到10×500、训练/验证/探针隔离的冻结规则。不得复制样本或悄悄改变数量。
2. **CCST 参数阻塞**：TA/TB 字符串未冻结，禁止编造并启动正式实验。
3. **Safe-FedLLM 方法变体**：论文将 sigmoid 线性形式称为 logistic regression；当前官方 commit 训练脚本为 PyTorch 单线性层+BCE/AdamW。实现同时明确标注“官方仓库线性 BCE 复现”与“任务书指定 sklearn Logistic Regression 变体”，不混称精度。
4. **CBDF 原文已取得**：已从 `mdpi-res.com` 获取 Electronics 14(10):2011 原文（DOI `10.3390/electronics14102011`），Algorithm 1 为最后一层权重更新欧氏距离、Top-K、有向计数和团检测；正文双向互选与伪代码任一有向阈值存在差异。新代码显式提供两种命名规则，默认 Algorithm 1。未发现经核实的官方实现，只能标为 faithful reimplementation。
5. **独立安全评价器未冻结**：Rule 与 BeaverTails 标签可作辅助，不足以单独构成独立语义安全评价；Kimi/GLM 不能作为唯一评价器。
6. **API 凭据未提供**：交互 SSH 环境中 Moonshot/智谱相关 API key 均未设置；模型列表接口未认证返回401。必须核实正式模型 ID 与价格，不能猜测或模拟。
7. **原始权重上下文**：后续已完成单adapter及received global的无损十进制容量预检，合计228,352,147个指令前Qwen tokens；服务商上下文未验证，Kimi/GLM真实分类仍NOT_RUN，详见基线报告。
8. 任务书“每轮保存历史客户端更新”与完整30轮×10×~20 MB存储的需求，以及中途逐步保存 optimizer/adapter 的IO成本，须通过短程真实测试测量。
9. FedAvg 对原始 A/B 参数平均不等于有效更新平均；新代码明确保留这一协议语义，检测使用 `alpha/r*(BA-B_gA_g)`。
10. 旧结果与配置全部保留；新输出采用唯一 `run_id`、原子写入与哈希校验，任何配置/数据/模型/代码不一致均停止恢复。
