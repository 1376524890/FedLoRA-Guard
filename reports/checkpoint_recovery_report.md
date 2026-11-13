# 检查点与恢复验收（最小必要范围）

更新时间：2026-11-12。按用户最新优先级，本报告记录实际状态，不继续围绕逐状态一致性扩展开发；重点转向真实训练、投毒、聚合和评价。

## 已实现

- 客户端每个优化步原子保存 LoRA、接收的全局 adapter、AdamW、scheduler、样本位置、已消费样本前缀、loss累计、投毒曝光及RNG。
- 每轮保存全局 adapter、客户端上传、聚合状态、参与列表、历史、检测摘要、RNG及所有文件SHA-256。
- 新 run 保存配置哈希、数据SHA、攻击派生视图SHA、clean源SHA、代码指纹和环境。
- 恢复校验模型/配置/数据/代码/攻击视图、checkpoint manifest和文件哈希；不连续轮次、哈希不匹配、篡改均拒绝。
- DataLoader只遍历未消费的计划后缀，并逐样本核对预期ID，避免恢复时从头重放。
- clean 与 attack tokenization cache 分别 keyed；攻击视图按轮选择。
- round-0初始adapter独立保存；不把它冒充独立clean反事实。

## 真实失败现场与状态

- `dev_smoke_killed_s17_v6` 保留实际 SIGKILL/EOFError 失败现场；没有 completed summary，因此**不宣称中途 kill/resume 已完整通过，也不宣称 bitwise GO**。
- `dev_smoke_reference_s17_v6` 是已完成的两轮clean轨迹，不能代替中断恢复验收。
- `dev_badnet_g4_temporal_s17_v1` 在第6轮真实训练60/60客户端任务后，检测阶段因dtype/None错误失败；没有完整round-6 checkpoint。该现场保留，不作为完成轨迹。
- `dev_badnet_g4_temporal_s17_v2` 在发现CBDF-LoRA观测/置换统计定义不一致后主动停止。旧progress若仍显示running，不是进程仍运行的证据；现场与旧输出均保留。
- 统计修正后以新 run `dev_badnet_g4_temporal_s17_v3` 复测，不覆盖旧结果。

## 结论

原子保存与严格一致性拒绝路径存在，且经过纯计算测试；**完整真实中途进程中断—恢复—最终状态对照仍为 FAIL / 未闭合**。整体正式矩阵不因此自动放行，也不在当前开发任务继续反复修补该单项。

正式运行前应至少完成一次小范围、明确run ID的中途kill、resume与独立不中断轨迹的最终adapter对比，再决定是否要求逐步bitwise；不能用已完成clean run或round-0代替此验收。
