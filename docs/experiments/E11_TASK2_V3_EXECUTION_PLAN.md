# E11 Task2 V3.1 执行说明

**正式执行依据**：[E11_TASK2_CORE_EXPERIMENT_PLAN.md](E11_TASK2_CORE_EXPERIMENT_PLAN.md)  
**Agent 提示词**：[E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md](prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md)  
**配置**：configs/e11/task2_core_v3.yaml  
**运行标识**：task2_v3_dolly_task1_target_s17  
**执行分支**：experiment/e11-task2-v3-baseline-preserving-20261126

## 执行原则

Task1 已经完成目标模型训练。Task2 直接读取 Task1 的30轮目标模型检查点、初始化适配器和最终适配器，不再重新训练 U 目标模型。

U 是由 Task1 目标检查点和九个真实正常客户端上传反算得到的单客户端等效上传。A、B 使用同一目标检查点、同一全局起点和八个正常背景客户端上传构造。

研究成功必须同时满足：

1. A、B 单独参与时目标行为弱。
2. A、B 联合参与时目标行为强，且连续评分交互项显著为正。
3. U 能被至少一种常规单客户端检测器识别。
4. A、B 单独检出率和 A+B 组合检出率达到正式检测规避门槛。
5. A、B 的有效更新、A 因子变化、B 因子变化分别接近正常客户端范围。

不要求 S-AB 与 U 的模型参数逐元素相等。数学因子关系用于检查构造正确性，行为和检测门槛用于判定研究结果。

## 执行阶段

1. P0：核验 Task1 三条 Dolly 目标轨迹、普通数据、旧 N 正常轨迹和独立锁定确认集。
2. P1：读取 Task1 目标检查点；从相同目标全局起点训练真实正常客户端上传。
3. P2：使用 s42、s2026 开发集执行225组 lambda 参数筛选、行为预检和检测规避预检。
4. P3：冻结参数，在 s17 留出系列上执行30轮同起点四分支验证。
5. P4：执行 S-A、S-B 各自30轮的持续性训练。
6. P5：完成 Norm、Cosine、FoolsGold、CBDF、Multi-Krum、Trimmed Mean 和统计分析。
7. P6：运行 Task2 独立锁定确认集一次，生成最终报告。

## 执行入口

~~~bash
bash scripts/run_e11_task2_core_v3.sh --status
bash scripts/run_e11_task2_core_v3.sh --stage P0
bash scripts/run_e11_task2_core_v3.sh --resume
bash scripts/run_e11_task2_core_v3.sh --all
~~~

当前脚本和训练代码必须与本文件及正式执行方案保持一致。Agent 先完成入口、数学实现、阶段状态机和测试，再启动 GPU 工作。

## 阶段交付

每个阶段都必须保存：

- run_state.json。
- 实际输入清单和 SHA256。
- 配置哈希和代码提交。
- 逐轮真实上传和模型检查点。
- 阶段指标 JSON。
- 阶段验收 Markdown。
- 阶段失败原因（如有）。

恢复时核验文件内容和哈希，不得仅根据文件存在判断阶段完成。所有阶段自动推进；任意硬门失败时保存证据并停止依赖该阶段的后续工作。
