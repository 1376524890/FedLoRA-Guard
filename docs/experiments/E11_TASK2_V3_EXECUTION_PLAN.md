# E11 Task2 V3 执行说明

**当前版本**：V3.0  
**正式执行依据**：[E11_TASK2_CORE_EXPERIMENT_PLAN.md](E11_TASK2_CORE_EXPERIMENT_PLAN.md)  
**Agent 执行提示词**：[E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md](prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md)  
**运行标识**：task2_v3_dolly_s17_a5  
**执行分支**：experiment/e11-task2-v3-baseline-preserving-20261126

## 文档使用规则

Task2 的研究目标、数据生成、训练配置、数学公式、参数筛选、行为门槛、检测协议、统计方法和阶段验收，全部以 E11_TASK2_CORE_EXPERIMENT_PLAN.md 为准。

本文件只提供执行入口和文档索引，不单独定义实验参数。Agent 不得从历史 Task2 文档、旧配置或旧训练产物中选择与正式执行依据冲突的设置。

## 执行顺序

1. 阅读正式执行方案的全部章节。
2. 阅读 Agent 提示词中的主控提示词和阶段提示词。
3. 核对 Git 分支、远程 HEAD 和工作区。
4. 按 P0 至 P6 状态机执行。
5. 每个阶段完成后保存真实输入、输出、哈希和验收结果。
6. 通过阶段门后自动进入下一阶段。
7. 失败时保留证据并停止，不得修改冻结门槛继续运行。

## 运行命令

~~~bash
bash scripts/run_e11_task2_core_v3.sh --status
bash scripts/run_e11_task2_core_v3.sh --stage P0
bash scripts/run_e11_task2_core_v3.sh --resume
bash scripts/run_e11_task2_core_v3.sh --all
~~~

执行入口必须支持状态查看、单阶段执行、断点恢复和完整无人值守运行。当前代码若尚未实现某个入口，Agent 必须先补齐实现和测试，再启动对应训练阶段。