# 数据验证报告（cyber1080）

日期：2026-11-12。执行目录：`/home/marktom/FedLoRA-Guard`。正式混合数据仍为 **NO-GO**。

## 实际原始输入

| 数据 | 固定来源/revision | 实测结果 | 状态 |
|---|---|---|---|
| Dolly-15K | `databricks/databricks-dolly-15k` / `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | 原始 SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`；14,824 条唯一有效记录 | 原始数据已验证 |
| PKU-SafeRLHF | `PKU-Alignment/PKU-SafeRLHF` / `9421ffafec3fa40a1f1a7d567b4d525079477ecb`；`data/Alpaca3-8B/train.jsonl` | SHA-256 `21c3b4f2572444e42d161f09166471d14208c9e153fcc0ec900ae05013e47d4c`；严格方向、非空且去重的配对 793 条 | 原始数量足够；正式分配未冻结 |
| BackdoorLLM | `BackdoorLLM/Backdoored_Dataset` / `cdb6a512ccb746913e7f3a853579d8c7830d3827` | `backdoor400_jailbreak_badnet.json`，302,066 bytes，400 条 `instruction/input/output`；本地 SHA-256 `cc41e753b48a866df15382664ccf306c228252d7860e1822eec6277695c8a95e` | 文件已下载；来源对象、license、clean 对照待核验 |
| BeaverTails Evaluation | `PKU-Alignment/BeaverTails-Evaluation` / `a93205d9ba6cc9dc398a34466da3602abddd2d40` | `v1/test.jsonl.xz` SHA-256 `6bd71b3136aa1a467a76dc300054513a2002608f304a7d30b1f5c116a2773e87`；700 行 | 可读；不替代独立语义安全评价器 |
| IFEval | google-research / `f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81` | 官方输入 541 条，SHA-256 `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`；固定 evaluator 文件、requirements、LICENSE 已取得 | 数据与 evaluator 已准备 |

证据：`.work/data_validation_current.json`、`references/instruction_following_eval/manifest.json`、各原始文件。

PKU 有 4 条方向满足条件但 safe response 为空的记录，源行 55、12498、13735、18101，均排除而不填补。完整 safe/unsafe 都符合 256-token 限制的配对为 **738** 条。没有截断、复制或修改源回答。

Backdoor 本地 SHA-1 为 `a2e04f796f409893e2d4c8d68013558667ac1098`，与先前元数据 oid `4a858e92416f5d3845f1f2e29d7773735759541b` 不一致，保留待核验。镜像后续 README/clean 对照请求遇到 503/timeout，未将失败响应当作有效文件。

## Dolly 工程数据（不是正式混合数据）

- `data/processed/dolly_smoke_len256_200/records.jsonl`：200 条训练（10×20）、256 公开探针、1000 clean test，共 1456 个唯一 ID。
- 使用真实 Qwen tokenizer 检查完整 chat prompt、response、EOS；只选择不超过 256 token 的记录，未截断。
- clean records SHA-256：`f8ffc4b30fe1489cf3f324192f603816e61f9a896938596997b2584fca7ad9c5`。
- 该入口只用于真实训练链路；不能替代正式 4200 Dolly + 400 Backdoor + 400 PKU 的分配规则。

## BadNet 工程派生视图

- `data/processed/dolly_badnet_g4_smoke/records.jsonl`，SHA-256 `4c884d005703d5134b5125410d7a729ef37981172a77b96222f35beabf83368d`。
- G4：`client_02`、`client_04`、`client_06`、`client_08`，各 20 条，合计 80 条替换记录。
- 源记录 1456，投毒视图仍 1456；非训练记录不携带 client 或污染标记。
- 投毒记录保存原/新 prompt、原/新 response、插入位置、trigger、target、client；全部完整符合 256 token。
- 这是冻结 BadNet 机制的 G4 工程视图，不是对 BackdoorLLM 原始 `BadMagic` 数据的替换或混合分配。

## PKU 工程派生视图

- `data/processed/pku_smoke_len256_200/`：200 训练 + 16 公开探针 + 16 clean test，共 232 条，规模明确缩小，**不冒充正式 1000 clean test**。
- clean records SHA-256：`218fdffef33310706cd9704734520f79c7672c62ecaf0567847b1e385cf48e9c`。
- attack records SHA-256：`45677e45c7363f3607319350c7806418934e9d10b29661af6072e29e2d15fcae`。
- `client_06`、`client_08` 各 5 条将 safe response 替换为同一 prompt 的 unsafe response；共 10 条。提示和配对关系保持不变。
- 初次尝试 1456 条 PKU-only 隔离数据因只有 738 条完整候选而明确失败；没有复制补足。

## 正式数据阻塞

`mixed_source_allocation: null`。4200 + 400 + 400 = 5000 只是总量，未定义各客户端来源分配，也未定义混合条件下 validation/public probe/clean test 的来源分配。代码拒绝猜测。必须冻结这些规则，并完成 Backdoor 来源/license 核验后，才能重新评估正式数据 GO。

旧 `data/data_validation.json` 保留；新校验写入 `.work/data_validation_current.json`，没有覆盖历史文件。
