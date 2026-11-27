"""E11 Task2 协议实现（唯一依据：docs/experiments/E11_TASK2_DESIGN.md）。

本包严格按设计文档 §1–§10 实现：Task1 T0 起步的正常联邦轨迹、从当前分支全局重训的 U、
00/U/10/01/11 五条持续轨迹与同轮影子对照、λ 交叉拆分、四分支交互项、检测与聚合防御。
所有服务器聚合一律调用仓库真实 ``src.fed.aggregate.fedavg``。
"""
