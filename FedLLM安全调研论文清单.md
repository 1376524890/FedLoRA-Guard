# FedLLM 安全调研：论文清单与阅读路线

> 调研主题：面向联邦 LoRA 微调的权重空间后门检测与修复
> 整理日期：2026-11-09（第一轮）；同日第二轮核实补全 GAP、确认 ProtegoFed 作者栏、PADBench 发布细节与 PEFTGuard 发表信息 ｜ 所有 arXiv 编号与 GitHub 仓库均已逐一核实（仓库可用性经 API 验证返回 200）

---

## 总览表

| # | 论文（简称） | 正式标题 | 发表 | arXiv | 代码 | 代码可用性 |
|---|---|---|---|---|---|---|
| A1 | OpenFedLLM | OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning | KDD 2024 | [2402.06954](https://arxiv.org/abs/2402.06954) | [rui-ye/OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) | ✅ 官方代码 |
| A2 | FedIT/Shepherd | Towards Building the Federated GPT: Federated Instruction Tuning | ICASSP 2024 | [2305.05644](https://arxiv.org/abs/2305.05644) | [JayZhang42/FederatedGPT-Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd) | ✅ 官方代码 |
| B1 | Safe-FedLLM | Safe-FedLLM: Delving into the Safety of Federated Large Language Models | arXiv 预印本（v5, 2026-05） | [2601.07177](https://arxiv.org/abs/2601.07177) | [dmqx/Safe-FedLLM](https://github.com/dmqx/Safe-FedLLM) | ✅ 官方代码 |
| B2 | BSNR | Revisiting Backdoor Threat in Federated Instruction Tuning from a Signal Aggregation Perspective | ICASSP 2026 | [2602.15671](https://arxiv.org/abs/2602.15671) | 未查到 | ❌ 需自行实现 |
| B3 | GAP | Low Rank Comes with Low Security: Gradient Assembly Poisoning Attacks against Distributed LoRA-based LLM Systems | arXiv 预印本（2026-01-02） | [2601.00566](https://arxiv.org/abs/2601.00566) | 未查到 | ❌ 需自行实现 |
| B4 | Lifespan | Watch Out for the Lifespan: Evaluating Backdoor Attacks Against Federated Model Adaptation | FPS 2025 | [2511.14406](https://arxiv.org/abs/2511.14406) | [gitlab.emse.fr/securityml/lora_backdoor_fl](https://gitlab.emse.fr/securityml/lora_backdoor_fl) | ✅ 官方代码（GitLab） |
| B5 | ProtegoFed | ProtegoFed: Backdoor-Free Federated Instruction Tuning with Interspersed Poisoned Data | arXiv 预印本（Work in progress） | [2603.00516](https://arxiv.org/abs/2603.00516) | [dongdongzhaoUP/ProtegoFed](https://github.com/dongdongzhaoUP/ProtegoFed) | ✅ 官方代码 |
| C1 | WeightWatch | Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs | arXiv 预印本（2025-08, CMU） | [2508.00161](https://arxiv.org/abs/2508.00161) | [fjzzq2002/WeightWatch](https://github.com/fjzzq2002/WeightWatch) | ✅ 官方代码 |
| C2 | LoRA 权重检测 | PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning | IEEE S&P 2025 | [2411.17453](https://arxiv.org/abs/2411.17453) | [Vincent-HKUSTGZ/PEFTGuard](https://github.com/Vincent-HKUSTGZ/PEFTGuard) | ✅ 官方代码 + PADBench |
| C3 | Haystack | The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers | arXiv 预印本（2026-02, Microsoft） | [2602.03085](https://arxiv.org/abs/2602.03085) | 未查到 | ❌ 需自行实现 |

---

## A. 环境搭建（第一优先）

### A1. OpenFedLLM —— 建议首选实验框架

- **正式标题**：OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning
- **作者**：Rui Ye, Wenhao Wang, Jingyi Chai, Dihan Li, Zexi Li, Yinda Xu, Yaxin Du, Yanfeng Wang, Siheng Chen（上海交通大学）
- **发表**：KDD 2024
- **arXiv**：[2402.06954](https://arxiv.org/abs/2402.06954)
- **代码**：https://github.com/rui-ye/OpenFedLLM（官方，已验证可用）
- **核心贡献**：研究友好的 FedLLM 一体化框架，覆盖联邦指令微调、联邦价值对齐、7 种 FL 算法、8 个训练数据集、30+ 评测指标；实验证明 FL 训练一致优于本地单独训练（金融基准上 Llama2-7B 联邦微调可显著超过 GPT-4）。
- **你的研究定位**：环境搭建第一优先；直接在其上注入恶意客户端与后门数据。

### A2. FedIT / Shepherd —— 最小可行实验与思路源头

- **正式标题**：Towards Building the Federated GPT: Federated Instruction Tuning
- **作者**：Jianyi Zhang, Saeed Vahidian, Martin Kuo, Chunyuan Li, Ruiyi Zhang, Tong Yu, Yufan Zhou, Guoyin Wang, Yiran Chen（Duke 大学 / Microsoft）
- **发表**：ICASSP 2024（arXiv v2 2024-01）
- **arXiv**：[2305.05644](https://arxiv.org/abs/2305.05644)
- **代码**：https://github.com/JayZhang42/FederatedGPT-Shepherd（官方，已验证可用）
- **核心贡献**：首次提出联邦指令微调（FedIT）范式，将 FL 用于 LLM 指令调优；开源 Shepherd 框架验证异构分布式指令数据 + FedAvg 聚合 LoRA 更新即可提升模型能力，为 FedLLM 奠定基础。
- **你的研究定位**：跑通最小 FedIT 实验；理解"客户端各自 LoRA → 服务器聚合"的最简形态。

**A 组建议阅读顺序**：OpenFedLLM（框架新、文档全、维护活跃，先跑 SFT 管线）→ FedIT/Shepherd（回到源头理解 FedIT 定义与 Shepherd 简化实现，便于复现最小实验）。

---

## B. 威胁模型与基线（第二优先）

### B1. Safe-FedLLM —— 最直接的防御基线

- **正式标题**：Safe-FedLLM: Delving into the Safety of Federated Large Language Models
- **作者**：Mingxiang Tao, Yu Tian, Wenxuan Tu, Yue Yang, Xue Yang, Xiangyan Tang
- **发表**：arXiv 预印本（v1 2026-01-12，v5 2026-05-30，未见正式会议信息）
- **arXiv**：[2601.07177](https://arxiv.org/abs/2601.07177)
- **代码**：https://github.com/dmqx/Safe-FedLLM（官方，已验证可用）
- **核心贡献**：从 LoRA 更新视角系统分析 FedLLM 攻击面；发现恶意客户端 LoRA 更新具有可区分的行为模式；提出基于轻量分类器的三级（Step-/Client-/Shadow-Level）探针防御框架，高恶意客户端比例下仍有效且几乎不影响训练速度。实验基线含 FedAvg、Multi-Krum、Trimmed Mean、FoolsGold、Residual、DnC、FLAME、LASA。
- **你的研究定位**：最直接的防御基线；其"LoRA 更新 = 高维行为特征"的建模与你权重空间检测思路同源，可直接对比。

### B2. BSNR —— 低浓度投毒的动机来源

- **正式标题**：Revisiting Backdoor Threat in Federated Instruction Tuning from a Signal Aggregation Perspective
- **作者**：Haodong Zhao, Jinming Hu, Gongshen Liu（上海交通大学）
- **发表**：ICASSP 2026
- **arXiv**：[2602.15671](https://arxiv.org/abs/2602.15671)
- **代码**：未查到公开仓库 —— **需按论文思想自行实现**
- **核心贡献**：挑战"只有恶意客户端才有威胁"的传统假设，研究**良性客户端数据集中散布的低浓度投毒**（自然触发器 + 对抗注入触发器两类）；提出信号聚合建模与 BSNR（Backdoor Signal-to-Noise Ratio）指标刻画分布式后门信号的上升-峰值-衰减动态；实验显示 <10% 投毒数据分布到各客户端即可达 85%+ ASR，且 Krum/FreqFed/FoundationFL 等现有防御在受影响客户端比例 >0.5 时失效。
- **你的研究定位**：低浓度投毒动机来源；你的权重空间检测正面向"现有防御失效"的空档。

### B3. GAP —— 直接攻击"分别上传 A、B"盲区的攻击方法

- **正式标题**：Low Rank Comes with Low Security: Gradient Assembly Poisoning Attacks against Distributed LoRA-based LLM Systems
- **作者**：Yueyan Dong, Minghui Xu, Qin Hu, Yinhao Xiao, Qi Luo, Yechao Zhang, Yue Zhang, Xiuzhen Cheng（华中科技大学 等）
- **发表**：arXiv 预印本（2026-01-02 提交，cs.CR）
- **arXiv**：[2601.00566](https://arxiv.org/abs/2601.00566)
- **代码**：未查到公开仓库（GitHub 精确搜索 0 结果）—— **需按论文思想自行实现**
- **核心贡献**：指出联邦 LoRA 框架（如 FedIT）的系统性盲区：客户端分别提交 A、B 矩阵，只有乘积 AB 决定模型更新，但乘积本身从未被直接验证。GAP 构造"单独看均良性、乘积却恶意"的 A/B 矩阵对，无需训练数据、无需客户端间协同即可绕过标准异常检测器。论文识别了 LoRA 联邦系统的 4 类系统性漏洞，在 LLaMA、ChatGLM、GPT-2 上验证：BLEU 最多降 14.5%，事实/语法错误增加 800%+，同时保持 92.6% 长回复长度（输出表面流畅，极难察觉）。
- **你的研究定位**：需关注乘积 ΔW —— 这是"必须检测乘积而非分量"的最强动机论文；你的权重空间检测若在聚合后的 ΔW=AB 上做谱分析，恰好堵住 GAP 利用的盲区，可作为直接针对性防御。
- **备注**：与 Safe-FedLLM 同期提交（2026-01），互相未引用属正常；不在后者的基线列表中不代表不重要——它攻击的是"更新构造层"，而 Safe-FedLLM 防的是"行为异常层"。

### B4. Watch Out for the Lifespan —— 后门持久性与 LoRA 秩

- **正式标题**：Watch Out for the Lifespan: Evaluating Backdoor Attacks Against Federated Model Adaptation
- **作者**：Bastien Vuillod, Pierre-Alain Moellic, Jean-Max Dutertre（Mines Saint-Étienne，法国）
- **发表**：FPS 2025（ Foundations & Practice of Security）
- **arXiv**：[2511.14406](https://arxiv.org/abs/2511.14406)
- **代码**：https://gitlab.emse.fr/securityml/lora_backdoor_fl（官方，GitLab 而非 GitHub）
- **核心贡献**：首个分析 LoRA 对 FL 模型适配后门攻击影响的工作；聚焦**后门生命周期（lifespan）**；关键发现：最优注入的后门在 **LoRA 秩越低时攻击后持久性越长**；同时指出该方向攻击评估方法学缺陷并给出更公平的评估协议。
- **你的研究定位**：需关注乘积 ΔW=BA —— LoRA 秩（即 ΔW 谱结构）直接决定后门残留时长，是你谱特征检测的机理依据。

### B5. ProtegoFed —— 分散投毒的样本级防御

- **正式标题**：ProtegoFed: Backdoor-Free Federated Instruction Tuning with Interspersed Poisoned Data
- **作者**：Haodong Zhao, Jinming Hu, Zhaomin Wu, Zongru Wu, Wei Du, Junyi Hou, Caibei Zhao, Zhuosheng Zhang, Bingsheng He, Gongshen Liu（上海交通大学 等）
- **发表**：arXiv 预印本（2026-02-28，标注 Work in progress）
- **arXiv**：[2603.00516](https://arxiv.org/abs/2603.00516)
- **代码**：https://github.com/dongdongzhaoUP/ProtegoFed（官方，已验证可用）
- **核心贡献**：面向"投毒样本散布于全部客户端（含良性客户端）"场景的首个 backdoor-free FIT 框架；用**频域梯度特征**区分投毒样本 + 全局二次聚类实现跨客户端协同识别；4 个数据集上检出 92%–100% 投毒样本、ASR 降至接近零。
- **与 BSNR 的关系（已核实）**：作者栏完全确认两文共享前三作者 Haodong Zhao、Jinming Hu（BSNR 为 Zhao/Hu/Liu 三人，ProtegoFed 为 Zhao/Hu 领衔的十人团队，Gongshen Liu 也在列）——**同一团队（上海交大 Gongshen Liu 组）**，ProtegoFed 是 BSNR 威胁模型的"官方解法"。
- **⚠️ 使用提示**：① 可利用——其实验设置很可能沿用 BSNR 的攻击设定，复现一套可同时得到攻击/防御对照；② 需小心——同团队"攻击+防御"组合的实验配置可能对自家方法有利，自建评测时应用独立设置交叉验证，不要照搬其配置。
- **你的研究定位**：样本级（数据空间）防御代表；与你的权重空间检测形成互补对照——它挡得住分散投毒，但作用于数据/梯度而非聚合后的 ΔW。

**B 组建议阅读顺序**：Safe-FedLLM（先立防御基线，代码全）→ **GAP（攻击"分别上传 A/B"盲区，与你的乘积 ΔW 思路直接对立统一，早读）** → BSNR（确立低浓度投毒威胁模型）→ ProtegoFed（同团队给出的数据空间解法，对照阅读，注意独立交叉验证）→ Lifespan（LoRA 秩与后门持久性，衔接 C 组谱分析）。

---

## C. 权重空间检测灵感（方法来源）

### C1. WeightWatch —— 方法灵感来源

- **正式标题**：Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs
- **作者**：Ziqian Zhong, Aditi Raghunathan（Carnegie Mellon University）
- **发表**：arXiv 预印本（2025-08，后经修订）
- **arXiv**：[2508.00161](https://arxiv.org/abs/2508.00161)
- **代码**：https://github.com/fjzzq2002/WeightWatch（官方，已验证可用；项目页 https://fjzzq2002.github.io/WeightWatch）
- **核心贡献**：提出**直接解读权重而非激活**的无监督、免数据方法：对"微调模型 − 基座模型"的权重差 ΔW 做 SVD，其顶部奇异向量（"行为向量"）对应微调新习得行为；推理时监控激活在这些方向上的余弦相似度即可高精度检出后门触发（13 个后门模型上拦截 44%–100%，FPR <1.2%，9/13 模型 ASR 清零），并可检测/恢复 unlearning 内容、审计商用模型微调重点。
- **你的研究定位**：方法灵感来源、谱特征基线 —— "ΔW 顶部奇异方向 ≈ 新行为（含后门）"是你整个课题的核心假设；联邦场景下 ΔW 即聚合 LoRA 乘积 BA，可自然迁移。

### C2. LoRA 权重检测 —— PEFTGuard（按你的备注推断匹配）

- **正式标题**：PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning
- **作者**：Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang（港科大广州、清华、国防科大 等）
- **发表**：**IEEE S&P 2025**（证据链：① arXiv 页面官方 journal reference 字段标注 "2025 IEEE Symposium on Security and Privacy (SP)"；② DOI 10.1109/SP61157.2025.00161 可解析至 IEEE Xplore 会议出版物页；③ DOI 关联页码 1620–1638）
- **arXiv**：[2411.17453](https://arxiv.org/abs/2411.17453)
- **代码**：https://github.com/Vincent-HKUSTGZ/PEFTGuard（官方，已验证可用）
- **PADBench（已确认公开）**：https://huggingface.co/datasets/Vincent-HKUSTGZ/PADBench
  - **注意**：HF 页当前实际托管的是 `finish_transfer_rank` 文件夹（约 32 个子目录的适配器 checkpoint），完整 13,300 适配器规模以论文原文为准；HF 卡片信息较简略，使用前建议对照论文附录或联系作者（Zhen Sun）。
  - **覆盖维度（从目录名解析）**：
    - 基座模型：Llama2-7B（主力）、Llama2-13B、Llama3-8B、ChatGLM6B、Flan-T5-XL、Qwen1.5-7B、Qwen2-VL、RoBERTa-base、ViT
    - LoRA 秩：8 / 16 / 32 / 64 / 128 / 256 / 512 / 1024 / 2048（toxic_backdoors_hard 系列全秩扫描）
    - PEFT 方法：LoRA、AdaLoRA、DoRA、QLoRA、LoRA+
    - 攻击：InsertSent、RIPPLES、Syntactic、StyleBkd、toxic backdoors（alpaca / hard）
    - 任务：AG_News、IMDB（分类）、SQuAD v2（问答）、toxic 指令（生成）
    - 目标模块后缀 `_qv` 提示主要针对 q/v 投影矩阵
- **方法结构**：对自注意力权重矩阵做特征变换后送元分类器（仓库含 train.py 与 contrast_train.py，暗示对比学习训练；预训练 checkpoint 按需向作者索取；README 无完整安装文档，依赖需从代码推断）
- **修复实验**：作者对比 SFT、DPO、Fine-mixing，发现 **Fine-mixing 修复效果最佳**
- **核心贡献**：首个面向 PEFT 适配器（LoRA 等）的后门检测框架；检测多数场景接近 100% 准确，零样本迁移到未见攻击/PEFT 方法/适配器秩；含自适应攻击实验。
- **你的研究定位**：LoRA 权重检测的当前 SOTA 与基准来源；**必须纳入检测基线**。
- **⚠️ 关键研究问题（本调研新识别）**：PADBench 场景是"开源平台共享的适配器"，而联邦场景的客户端更新是**同一全局适配器出发、本地训练步数少、数据非独立同分布**——两者分布差异显著。**"在 PADBench 上训练的 PEFTGuard 能否迁移到联邦客户端更新上"** 是一个简单、可检验、无论结果正负均可成文的研究问题，建议作为你实验设计的第三层（PADBench 复现 → 自建联邦评测 → 跨域迁移）。

### C3. Trigger in the Haystack —— 触发器提取式扫描

- **正式标题**：The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers
- **作者**：Blake Bullwinkel, Giorgio Severi, Keegan Hines, Amanda Minnich, Ram Shankar Siva Kumar, Yonatan Zunger（Microsoft）
- **发表**：arXiv 预印本（2026-02-03）
- **arXiv**：[2602.03085](https://arxiv.org/abs/2602.03085)
- **代码**：未查到公开仓库 —— **需按论文思想自行实现**
- **核心贡献**：面向 sleeper-agent 式后门（sleeper agents 会记忆投毒数据）的实用扫描器：① 用记忆提取技术泄露后门样本；② 利用触发词出现时输出分布与注意力头的特异模式；无需预知触发器或目标行为、仅需推理操作即可恢复可用触发器，覆盖多种后门场景与微调方法，可直接嵌入部署前审计流程。
- **你的研究定位**：方法灵感来源（输出分布/注意力信号一侧）；与你的权重谱方法互补——它证明"行为侧信号"足以复原触发器，可作为联合检测的对照或增强信号。

**C 组建议阅读顺序**：WeightWatch（确立 ΔW 谱方向假设，最优先精读）→ PEFTGuard（工程化检测 + 基准，承接假设落地）→ Trigger in the Haystack（行为侧信号，拓展检测面）。

---

## 需自行实现代码的论文

| 论文 | 原因 | 实现要点提示 |
|---|---|---|
| B2 (BSNR) | 无公开代码 | 在 OpenFedLLM 上构造"低浓度投毒散布于良性客户端"数据流；实现 BSNR 指标追踪信号聚动态 |
| B3 (GAP) | 无公开代码 | 核心是构造"单独良性、乘积恶意"的 A/B 矩阵对（论文识别了 4 类系统性漏洞可利用）；实现后即可作为你乘积 ΔW 检测的直接靶标 |
| C3 (Trigger in the Haystack) | 无公开代码 | 记忆提取（提示诱导泄露）+ 触发词出现时的输出分布/注意力统计检测器 |

其余 7 篇均有官方代码（OpenFedLLM、Shepherd、Safe-FedLLM、Lifespan、ProtegoFed、WeightWatch、PEFTGuard），可基于其复现改造。

---

## 基线与评测设计（依据本轮核实更新）

### 检测基线清单
1. **PEFTGuard**（S&P 2025，权重空间元分类器，最标准基线，须纳入）
2. **Safe-FedLLM**（LoRA 更新 = 行为特征 + 轻量分类器，联邦设定下的直接对照）
3. **WeightWatch 风格 SVD**（ΔW 顶部奇异向量 + 激活余弦监控，无监督）
4. **LoRA 谱统计**（奇异值分布、有效秩等轻量统计特征）
5. （可选）**LLM 审计 agent**（行为侧信号，参考 Trigger in the Haystack 思路）

### 修复基线清单
- **Fine-mixing**（PEFTGuard 论文验证的最有效修复方法，须纳入）
- SFT / DPO（PEFTGuard 论文中的对照修复方法）

### 新颖性定位（重要修正）
"仅看权重检测 LoRA 后门"已有 PEFTGuard 做透并配上体系化基准（S&P 2025），**不能再作为创新点**。创新点只能落在：
- **联邦设定**（跨客户端、多轮聚合下的检测）
- **低浓度分散投毒**（BSNR 揭示、现有防御失效的场景）
- **乘积 ΔW 视角**（GAP 证明分量检测有盲区，乘积层检测恰好补位）
- **LLM 审计 agent + 修复闭环**（检测 → 定位 → 修复的完整链路）

### 三层实验路线
1. **第一层**：在 PADBench 上复现 PEFTGuard，确认实现正确性（对齐论文数字）。
2. **第二层**：在自建联邦数据（OpenFedLLM/Shepherd + BSNR 攻击设定 + GAP 攻击）上评测各基线与你的方法；评测配置独立设计，不照搬 BSNR/ProtegoFed 的设置，交叉验证。
3. **第三层**：PADBench → 联邦的**跨域迁移实验**（在 PADBench 训练的 PEFTGuard 能否迁移到联邦客户端更新上）——这是本轮调研识别出的新研究问题，简单可检验，正负结果均可成文。

---

## 总体执行路线建议

1. **第 1 步（A 组）**：OpenFedLLM 跑通联邦指令微调 → Shepherd 理解最小 FedIT 形态。
2. **第 2 步（B 组）**：Safe-FedLLM 建立防御基线与威胁实验协议；GAP（自行实现）构造乘积层攻击作为直接靶标；BSNR 构造低浓度投毒威胁模型（自行实现）；Lifespan 注意 LoRA 秩与后门持久性的关系（ΔW 谱结构证据）。
3. **第 3 步（C 组）**：WeightWatch 的 ΔW 奇异向量分析迁移到联邦聚合 LoRA（ΔW=Σ·BA）；按三层实验路线落地 PEFTGuard 复现 → 自建联邦评测 → 跨域迁移；Trigger in the Haystack 提供行为侧补充信号。
4. **已全部核实**：10 篇论文条目完整（GAP 已于 2026-11-09 第二轮核实补全）。
