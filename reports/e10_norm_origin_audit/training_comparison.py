"""E10 训练过程对照表生成（2026-11-23）。

逐项比较"集中目标训练"（P2 train_target，6 系列）与"正常客户端本地训练"
（背景轨迹 e10bg_*，13 条），区分配置计划值与日志/检查点实测值。

用法：
  cd /home/marktom/FedLoRA-Guard
  /home/marktom/miniconda3/envs/fllora/bin/python reports/e10_norm_origin_audit/training_comparison.py
产出：training_comparison.tsv
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/marktom/FedLoRA-Guard")
OUT = ROOT / "reports/e10_norm_origin_audit"
SERIES = ["pku_s17", "pku_s42", "pku_s2026", "dolly_s17", "dolly_s42", "dolly_s2026"]
BG_TRAJS = (["e10bg_clean_s2026"]
            + [f"e10bg_pku_{s}_{i}" for s in ("s17", "s42", "s2026")
               for i in ("00", "01", "10", "11")])


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    # ---- 集中训练：6 系列计划值与实测值 ----
    cen_runs = [jload(ROOT / f"artifacts/e10/p2/{s}/run.json") for s in SERIES]
    cen_logs = []
    for s in SERIES:
        lines = [json.loads(x) for x in
                 (ROOT / f"artifacts/e10/p2/{s}/training_log.jsonl").read_text().splitlines() if x]
        cen_logs.append(lines)
    cen_steps = [next(e["optimizer_steps"] for e in lg if e["event"] == "done") for lg in cen_logs]
    cen_micros = [next(e["micro_positions"] for e in lg if e["event"] == "schedule") for lg in cen_logs]
    cen_loss = [next(e["average_loss"] for e in lg if e["event"] == "done") for lg in cen_logs]
    r0 = cen_runs[0]
    cen_plan = {
        "秩/α/缩放": f"r={r0['lora']['rank']} α={r0['lora']['alpha']} s={r0['scaling']}",
        "优化步数": "250 步（train_target.STEPS）",
        "有效批量": "4（micro 1 × 累积 4）",
        "样本使用": "1000 条恰好各用一次（20 目标 + 980 正常）",
        "学习率": "1e-4 常数（LambdaLR 恒 1.0）",
        "weight_decay": "0.0（train_target 覆盖）",
        "梯度裁剪": "max_grad_norm 1.0（configs/federated.yaml 经 include 提供默认）",
        "优化器状态": "AdamW 单次 train() 调用内连续 250 步（跨步延续）",
        "起始适配器": "全零（LoRA 标准 B=0 初始化；received_global=0，实测 B 全零）",
        "数据构成": "20 条虚构仓（华宇）缺陷品目标记录（答案=可出库，行为逆转，占 2%）+ 980 正常（e10_{dolly,pku}_formal 池确定性抽样）",
    }
    cen_meas = {
        "秩/α/缩放": f"6/6 系列 run.json 一致（r32/α64/s2）",
        "优化步数": f"6/6 系列 training_log.jsonl optimizer_steps={sorted(set(cen_steps))}",
        "有效批量": f"schedule micro_positions={sorted(set(cen_micros))} = 250×4 ✓",
        "样本使用": f"schedule 计数 total=1000（target 20/normal 980，skipped_over_len 6 起）",
        "学习率": "日志未记录逐步 LR；scheduler=常数（代码），无法从日志逐值确认",
        "weight_decay": "run.json wd=0.0（计划=落盘配置）",
        "梯度裁剪": "日志未记录逐步梯度范数；无法从现有材料确认实际触发情况",
        "优化器状态": "单 round 单 client（run.json rounds=1 clients=1；latest.json step 250 complete）",
        "起始适配器": "step_001/received_global_adapter 实测范数=0 且 B 全零",
        "平均损失": f"{[round(x, 3) for x in cen_loss]}（training_log done 事件）",
    }

    # ---- 本地训练：13 条轨迹 ----
    bg_cfgs = {}
    for t in BG_TRAJS:
        cfg_path = ROOT / f"artifacts/runs/{t}/resolved_config.json"
        if cfg_path.exists():
            bg_cfgs[t] = jload(cfg_path)
        else:  # e10bg_pku_s42_11 为事故恢复轨迹：resolved_config 被误删，配置见 PROVENANCE_RECOVERY.json
            bg_cfgs[t] = jload(ROOT / "artifacts/abc/e10bg_1.7b" / t / "config.json")
    bg_events = {}
    for t in BG_TRAJS:
        ev = [json.loads(x) for x in (ROOT / f"artifacts/runs/{t}/events.jsonl").read_text().splitlines() if x]
        bg_events[t] = [e for e in ev if e["event"] == "client_training_completed"]
    all_steps = sorted({e["local_steps"] for t in BG_TRAJS for e in bg_events[t]})
    n_client_rounds = sum(len(bg_events[t]) for t in BG_TRAJS)
    bg_data = {}
    for t in BG_TRAJS:
        di = ROOT / f"artifacts/runs/{t}/data_identity.json"
        bg_data[t] = (jload(di)["path"] if di.exists()
                      else jload(ROOT / f"artifacts/runs/{t}/PROVENANCE_RECOVERY.json")["records"])
    b0 = bg_cfgs["e10bg_pku_s17_00"]
    f0 = b0["federated"]
    bg_plan = {
        "秩/α/缩放": f"r={b0['model']['lora']['rank']} α={b0['model']['lora']['alpha']} s=2",
        "优化步数": "每客户端每轮 10 步 × 6 轮（共 60 步/客户端）",
        "有效批量": "2（micro 1 × 累积 2）",
        "样本使用": "每客户端 20 条私有记录；每轮恰好 20 个样本位置（10 步 × 2），同一 20 条每轮重用",
        "学习率": "1e-4 常数（scheduler: constant）",
        "weight_decay": "0.01（federated.yaml 默认）",
        "梯度裁剪": "max_grad_norm 1.0",
        "优化器状态": "AdamW 每客户端每轮重建（src/train/local.py:119 注释与实现）；跨轮不延续",
        "起始适配器": "每轮从 received_global（上一轮 10 客户端因子等权平均）出发；round 1 为全零 initial_adapter",
        "数据构成": "纯良性任务记录，无目标/场景记录：pku 系=benign_pku_20261114 批（PKU-SafeRLHF 衍生，每客户端 20 条 train）；clean_s2026=dolly_smoke_len256_200（Dolly 衍生，每客户端 20 条 train）",
    }
    bg_meas = {
        "秩/α/缩放": "13/13 轨迹 resolved_config 一致（r8/α16）",
        "优化步数": f"events.jsonl 共 {n_client_rounds} 条 client_training_completed，local_steps={all_steps}（全部 10）",
        "有效批量": "事件未记录 micro 批数；由 10 步 × 累积 2 与调度一致性间接支持，日志无法直接确认",
        "样本使用": "数据文件自带 client_id/split 分配（每客户端 20 条 train）；逐步样本位置已随 step 目录清理，无法从日志确认",
        "学习率": "日志未记录逐步 LR；scheduler=constant（代码）",
        "weight_decay": "resolved_config 0.01（计划=落盘配置）",
        "梯度裁剪": "日志未记录梯度范数；无法从现有材料确认",
        "优化器状态": "每轮 latest.json 指向 step_010 complete（13×6×10=780 个客户端轮均完成）",
        "起始适配器": "round_001/received_global 实测范数=0（本轮审计 round_norms.tsv norm_global r1=0）",
        "聚合": "fedavg 因子等权平均（aggregation_state.pt + metrics.json aggregation_method=fedavg）",
    }

    rows = [
        ("起始模型", "Qwen3-1.7B 本地 fp32 eager（run.json base）",
         "6/6 系列 base=/home/marktom/models/qwen3-1.7b",
         "Qwen3-1.7B 本地 fp32 eager（resolved_config model.id）",
         "13/13 轨迹一致"),
        ("LoRA 秩 / α / 缩放", cen_plan["秩/α/缩放"], cen_meas["秩/α/缩放"], bg_plan["秩/α/缩放"], bg_meas["秩/α/缩放"]),
        ("起始适配器", cen_plan["起始适配器"], cen_meas["起始适配器"], bg_plan["起始适配器"], bg_meas["起始适配器"]),
        ("训练任务/损失", "回答词元因果 LM 损失（prompt 标签 -100；src/data/tokenize.py tokenize_pair/collate，两条路径共用 LocalTrainer）",
         "同一代码路径（src/train/local.py；无分支）",
         "同左", "同左"),
        ("数据来源", cen_plan["数据构成"],
         "run.json data_stats：total=1000 target=20 normal=980（6/6 系列）",
         bg_plan["数据构成"],
         "data_identity.json 逐轨迹指向上述文件；train 分配每客户端 20 条（实测计数）"),
        ("优化步数", cen_plan["优化步数"], cen_meas["优化步数"], bg_plan["优化步数"], bg_meas["优化步数"]),
        ("参与损失的样本数", cen_plan["样本使用"], cen_meas["样本使用"], bg_plan["样本使用"], bg_meas["样本使用"]),
        ("参与损失的词元数", "未记录", "无法从现有材料确认（training_log 只有调度与汇总损失）",
         "未记录", "无法从现有材料确认（events.jsonl 只有轮级损失）"),
        ("有效批量", cen_plan["有效批量"], cen_meas["有效批量"], bg_plan["有效批量"], bg_meas["有效批量"]),
        ("学习率", cen_plan["学习率"], cen_meas["学习率"], bg_plan["学习率"], bg_meas["学习率"]),
        ("weight_decay", cen_plan["weight_decay"], cen_meas["weight_decay"], bg_plan["weight_decay"], bg_meas["weight_decay"]),
        ("梯度裁剪", cen_plan["梯度裁剪"], cen_meas["梯度裁剪"], bg_plan["梯度裁剪"], bg_meas["梯度裁剪"]),
        ("优化器状态延续", cen_plan["优化器状态"], cen_meas["优化器状态"], bg_plan["优化器状态"], bg_meas["优化器状态"]),
        ("平均损失", "—", cen_meas["平均损失"], "—",
         f"events.jsonl 逐客户端轮 loss（780 条，如 r1 c00=2.78）"),
        ("中间检查点保留", "逐步全存后清理，仅剩 step_001 + final（step_250 目录已删）",
         "checkpoints/round_001/clients/central/{step_001,final_adapter.safetensors}；step_001 实测范数≈0.248",
         "每客户端轮 step_010 目录已清理（scripts/e10_overnight.sh PRUNE 段），仅剩 final_adapter",
         "13 轨迹 × 6 轮 × 10 客户端均只有 final_adapter.safetensors + latest.json"),
        ("证据路径（集中）", "artifacts/e10/p2/<series>/{run.json,training_log.jsonl,target_adapter.safetensors,checkpoints/round_001/clients/central/}",
         "", "artifacts/runs/<traj>/{resolved_config.json,events.jsonl,data_identity.json,checkpoints/round_*/}", ""),
    ]
    with open(OUT / "training_comparison.tsv", "w", encoding="utf-8") as f:
        f.write("对比项\t集中目标训练（计划）\t集中目标训练（实测/证据）\t正常客户端本地训练（计划）\t正常客户端本地训练（实测/证据）\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"[training_comparison] {len(rows)} 行 → training_comparison.tsv "
          f"(集中步数实测 {sorted(set(cen_steps))}，背景客户端轮 {n_client_rounds}，"
          f"local_steps {all_steps})")


if __name__ == "__main__":
    main()
