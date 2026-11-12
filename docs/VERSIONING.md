# 版本管理规范

**适用：** FedLoRA-Guard 实验仓库（正式实验开始前一律 0.x）

## 1. 版本号（SemVer 0.x）

`0.MINOR.PATCH`

| 位 | 含义 | 示例 |
|---|---|---|
| MINOR | 阶段推进 / 新能力（新模块、新实验类型、配置格式变化） | 0.1.0 建仓；0.2.0 指标与计时修复 |
| PATCH | 不改行为的修复（bug fix、文档、脚本修正），或**同代码重跑** | 0.2.1 |

- 正式矩阵通过 GO 判定并首次完整执行 → 1.0.0。
- 兼容性破坏（配置格式、产物 schema 变化）也进 MINOR，并在 CHANGELOG 标注 BREAKING。

## 2. 分支模型

- `main`：始终可运行；只接受通过验证的提交（测试通过 + 至少一次真实运行）。
- `dev/<topic>`：日常开发分支（如 `dev/metrics-fix`）。
- 服务器运行只从 `main`（或显式 tag）checkout。

## 3. run_id 与产物

- `run_id = YYYYMMDD_<stage>_<scenario>_s<seed>`（小写，下划线分隔）。
- 例：`20261112_p0_timing_conc_s17`、`20261201_badnet_g2_s17`。
- 每次运行写 `runs/<run_id>/manifest.json`：

```json
{
  "run_id": "...",
  "git_commit": "<full sha>",
  "git_tag": "v0.2.0",
  "config_path": "configs/timing_v1.yaml",
  "config_snapshot": { ... },
  "environment": { "python": "...", "torch": "...", "gpus": [...] },
  "seed": 17,
  "started_at": "ISO-8601",
  "completed_at": "ISO-8601"
}
```

- 产物目录不进 git（.gitignore），归档脚本 `scripts/archive_run.sh` 打 tar 保留。
- 跨版本比较结果时必须核对 manifest 的 git_commit。

## 4. CHANGELOG

每次版本发布在 CHANGELOG.md 顶部追加：

```markdown
## [0.2.0] - 2026-11-12
### Added / Changed / Fixed / Verified
- ...
### Verification
- pytest N passed
- run_id=... （真实运行证据）
```

## 5. 数据与结果纪律（继承指引 §2）

1. 不覆盖旧 run 目录；run_id 全局唯一。
2. 每个实验参数变化必须体现在 config 快照，不允许口头配置。
3. 失败结果保留并记录原因；禁止删除。
4. 指标未定义时写 null + undefined_reason，不写 0。
5. 单元测试禁止 mock 训练/评价结果；数学测试允许合成矩阵。

## 6. 与服务器同步

- 本地 `/Users/didi/myWork/FedLoRA-Guard/experiment` 为开发主库（git）。
- 服务器 `cyber1080:~/FedLoRA-Guard` 为运行副本：`rsync` 同步 src/tests/configs/scripts，或 `git push` + 服务器 `git pull`（推荐，服务器仅作只读运行）。
- 服务器运行产物通过 `scripts/fetch_results.sh` 拉回本地 `runs/`（本地同样 gitignore）。
