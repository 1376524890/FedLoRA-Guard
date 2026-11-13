# 硬件与运行基准（cyber1080）

## 固定环境

- 主机：`cufewangyancyber`；3× NVIDIA GeForce GTX 1080 Ti，11264 MiB，compute capability 6.1。
- 实验解释器：`/home/marktom/miniconda3/envs/fllora/bin/python`，Python 3.11.18。
- Torch 2.5.1+cu118；Transformers 4.53.3；PEFT 0.16.0；Datasets 3.6.0；NumPy 1.26.4；CUDA runtime 11.8。
- 系统 `/usr/bin/python3` 是受 PEP 668 管理的另一解释器，没有在其中使用 `--break-system-packages`。
- NetworkX 曾误降为 3.4.2，随后立即恢复原已安装的 3.6.1；最终环境以各 run 的 `environment.json` 为准。

## 旧实测（保留，不作为新系统完成证据）

| 项目 | 实测 |
|---|---:|
| GPU0 FP32 step | 1.548590 s |
| GPU1 FP32 step | 1.574916 s |
| GPU2 FP32 step | 1.521072 s |
| 峰值 reserved | 3.716797 GiB |
| 峰值 allocated | 2.814908 GiB |
| 可训练 adapter 参数 | 5,046,272 |
| 三卡两轮旧计时总墙钟 | 330.322 s |
| 两轮墙钟 | 162.519 / 156.547 s |

旧计时为 25 优化步/客户端/轮；冻结协议为 10 步，不能直接混用吞吐量。

## 新系统真实短程 smoke

- run：`artifacts/runs/dev_smoke_uninterrupted_s17_v4/`。
- 真实 Qwen3-0.6B、FP32、七模块 LoRA、三张固定 GPU worker，每卡一个模型副本。
- round 1 完成 10 客户端×10 优化步；客户端耗时 35.80–52.33 秒（包含逐优化步保存和校验）。
- 客户端实测 loss 有有限值，所有 GPU 上传 adapter 已保存；round 1 原子 checkpoint manifest `complete=true`。
- 当前 run 在 round 1 保存后因日志函数重复 `round` 参数退出；恢复兼容过程另见 `checkpoint_recovery_report.md`。没有将该 run 标为两轮不中断基线。
- 运行中观测 GPU 显存约 3.6–4.3 GiB（`nvidia-smi` 时点值）；完整峰值见客户端结果/checkpoint。

## 新系统真实 FP32/FP16 基准（GPU1）

结果：`.work/precision_gpu1_real_v2.json`，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`，同一真实公开探针、microbatch=1、累积=2、AdamW、gradient checkpointing、10优化步；仅测吞吐，不是攻击训练。首次运行在CUDA初始化前重置峰值统计失败，保留旧错误；修正后以新路径重跑成功。

| dtype | load秒 | 平均优化步秒 | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|
| FP32 | 4.583937 | 1.706161 | 2.601692 | 2.679443 |
| FP16 | 3.322011 | 1.891648 | 1.484320 | 1.548553 |

GPU1单卡该实现下FP16更慢，但显存峰值更低；不外推为三卡线性加速。上述平均包含首步，逐步时间完整保存在JSON。

## 新系统三卡精度补齐（独立空闲GPU0/GPU2物理映射）

`.work/precision_gpu0_real_v1.json` 已完成：FP32 load=4.301834秒、平均步1.617605秒、allocated=2,793,954,304 bytes、reserved=2,877,292,544 bytes；FP16 load=2.964299秒、平均步1.826336秒、allocated=1,593,614,848 bytes、reserved=1,663,041,536 bytes。

物理GPU2首次用`CUDA_VISIBLE_DEVICES=1 --gpus 2`失败，原因是逻辑设备只有0却请求2；错误保留。修正后以`CUDA_VISIBLE_DEVICES=2 --gpus 0`重跑，`.work/precision_gpu2_real_v2.json` 已完成：FP32 load=4.433907秒、平均步1.589481秒、allocated=2,793,954,304 bytes、reserved=2,877,292,544 bytes；FP16 load=2.949307秒、平均步1.771797秒、allocated=1,593,614,848 bytes、reserved=1,663,041,536 bytes。导出证据显式保存physical_gpu_id=2、logical_gpu_id=0。运行均使用空闲物理卡，避免与生成评价争用。

| 物理GPU | FP32秒/步 | FP16秒/步 | FP32 allocated GiB | FP16 allocated GiB |
|---|---:|---:|---:|---:|
| 0 | 1.617605 | 1.826336 | 2.601692 | 1.484320 |
| 1 | 1.706161 | 1.891648 | 2.601692 | 1.484320 |
| 2 | 1.589481 | 1.771797 | 2.601692 | 1.484320 |

三卡均显示当前实现FP16较慢但显存更低。该测量不是正式攻击训练，也不外推完整30轮矩阵吞吐。

## 新系统完成的端到端真实训练

- 两轮G4：20/20客户端任务，每轮投毒曝光80。
- 两轮PKU：20/20客户端任务，每轮投毒曝光10；round 1/2为174.872988/207.817274秒，总耗时414.217508秒。
- 六轮G4检测v3：60/60客户端任务，6/6轮checkpoint，elapsed=1141.708255秒；最终digest `7abbcf2a0ab529888549e2914f59afb472aaf1ab9b221ee81acea6346674f147`。



单个 adapter 为 5,046,272 个 FP32 参数，约 20.24 MB 二进制；10 个约 201.8 MB。逐步 checkpoint 同时保存 client/global adapter、optimizer、scheduler、RNG 和哈希，I/O 成本不可忽略。30 轮完整矩阵的存储、训练墙钟和生成评价成本尚未完成基准，不给出虚假的精确 ETA。正式 110 轨迹未启动。
