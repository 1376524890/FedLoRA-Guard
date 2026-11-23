"""E11 最终报告生成（P6 末尾调用）：RESULTS.md / E11_FINAL_REPORT.md / run_manifest.json。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e11.analysis import norm_match_summary  # noqa: E402
from src.e11.runner import BRANCHES  # noqa: E402


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _r(x, n=3):
    return None if x is None else round(x, n)


def _fmt_w(w: dict) -> str:
    if not w or w.get("rate") is None:
        return "—"
    return f"{w['hits']}/{w['n']} ({w['rate']:.1%} [{_r(w['lower'])},{_r(w['upper'])}])"


def write_final_reports(e) -> None:
    cfg = e.cfg
    rep = e.rep("final")
    series_all = cfg["series"]
    base = _load(ROOT / f"reports/e11_p0_{e.run_id}/base_metrics.json")
    branch = _load(rep / "branch_metrics.json")
    inter = _load(rep / "interaction_metrics.json")
    det = _load(rep / "detector_metrics.json") if (rep / "detector_metrics.json").exists() else {}
    defense = _load(rep / "defense_metrics.json") if (rep / "defense_metrics.json").exists() else {}
    norms = {s: norm_match_summary(e.art / s, cfg) for s in series_all}

    lines = ["# E11 正式实验结果（RESULTS）", "",
             f"run_id: {e.run_id}　分支: {cfg['git_branch']}　种子: 17/42/2026 × Dolly/PKU", "",
             "## 基座模型（四仓库最终测试 + 闭集）", "",
             "| 指标 | 基座 |", "|---|---|"]
    for k in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "FPR_nanzhan",
              "Acc_normal", "Pair_exact"):
        lines.append(f"| {k} | {_fmt_w(base[k])} |")
    lines.append(f"| Acc_dolly_option | {_fmt_w(base['closedset']['option'])} |")
    lines.append(f"| Acc_dolly_strict | {_fmt_w(base['closedset']['strict'])} |")
    lines.append(f"| Target_margin(华宇缺陷) | {_r(base.get('margin_huayu_defective'))} |")

    for s in series_all:
        lines += ["", f"## {s}", "",
                  "| 分支 | 华宇放行 | 城东误放 | 北港误放 | 南站误放 | 正常品放行 | Pair_exact | 闭集option | 闭集strict | margin华宇 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b in BRANCHES:
            m = branch[s][b]
            lines.append(
                f"| {b} | {_fmt_w(m['Y_huayu_release'])} | {_fmt_w(m['FPR_chengdong'])} | "
                f"{_fmt_w(m['FPR_beigang'])} | {_fmt_w(m['FPR_nanzhan'])} | {_fmt_w(m['Acc_normal'])} | "
                f"{_fmt_w(m['Pair_exact'])} | {_fmt_w(m['closedset']['option'])} | "
                f"{_fmt_w(m['closedset']['strict'])} | {_r(m.get('margin_huayu_defective'))} |")
        iv = inter[s]
        lines += ["", f"交互量（商品配对 Bootstrap 95% CI）：I_H={_r(iv['I_H']['mean'])}"
                  f" [{_r(iv['I_H']['ci'][0])},{_r(iv['I_H']['ci'][1])}]，"
                  f"gamma_华宇={_r(iv['gamma_huayu']['mean'])}"
                  f" [{_r(iv['gamma_huayu']['ci'][0])},{_r(iv['gamma_huayu']['ci'][1])}]，"
                  f"Δ_A={_r(iv['delta_A']['mean'])}，Δ_B={_r(iv['delta_B']['mean'])}，"
                  f"Δ_AB={_r(iv['delta_AB']['mean'])}",
                  f"更新大小：中位比值={_r(norms[s]['median_ratio'])}"
                  f"（各分支 {json.dumps({k: _r(v) for k, v in norms[s]['per_branch_median_ratio'].items()}, ensure_ascii=False)}），"
                  f"落入良性带比例={_r(norms[s]['coverage_in_benign_band'])}，"
                  f"匹配通过={norms[s]['pass_ratio'] and norms[s]['pass_band']}"]

    if det:
        lines += ["", "## 客户端检测（种子留一校准，冻结阈值）", "",
                  "| 系列 | Norm(5%) P/R/FPR/F1 | Cosine 对 recall/良性对误报 | FoolsGold 冷启动 recall |",
                  "|---|---|---|---|"]
        for s in series_all:
            if s not in det:
                continue
            n = det[s].get("norm_fpr5", {})
            c = det[s].get("cosine_pair", {})
            f = det[s].get("foolsgold_cold", {})
            lines.append(f"| {s} | {_r(n.get('precision'))}/{_r(n.get('recall'))}/"
                         f"{_r(n.get('fpr'))}/{_r(n.get('f1'))} | {_r(c.get('recall'))}/"
                         f"{_r(c.get('benign_pair_fpr'))} | {_r(f.get('recall'))} |")
    if defense:
        lines += ["", "## 聚合防御（round 30）", "",
                  "| 系列 | 条件 | 防御 | 华宇放行 | 城东误放 | 北港误放 | 南站误放 | 正常品 | Pair_exact |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s in series_all:
            for key, v in defense.get(s, {}).items():
                cond, name = key.split("|")
                m = v["metrics"]
                lines.append(f"| {s} | {cond} | {name} | {_fmt_w(m['Y_huayu_release'])} | "
                             f"{_fmt_w(m['FPR_chengdong'])} | {_fmt_w(m['FPR_beigang'])} | "
                             f"{_fmt_w(m['FPR_nanzhan'])} | {_fmt_w(m['Acc_normal'])} | "
                             f"{_fmt_w(m['Pair_exact'])} |")

    # 最终验收
    fa = cfg["acceptance"]["final"]
    gate_rows = []
    for s in series_all:
        m = branch[s]
        y = {b: m[b]["Y_huayu_release"]["rate"] for b in BRANCHES}
        checks = {
            "single_huayu": all(y[b] <= fa["single_huayu_max"] for b in ("00n", "10n", "01n")),
            "fpr": all(m[b][f"FPR_{w}"]["rate"] <= fa["fpr_max"] for b in BRANCHES
                       for w in ("chengdong", "beigang", "nanzhan")),
            "joint_huayu": y["11"] >= fa["joint_huayu_min"],
            "normal": all(m[b]["Acc_normal"]["rate"] >= fa["normal_min"] for b in BRANCHES),
            "closedset": all(m["00n"]["closedset"]["option"]["rate"]
                             - m[b]["closedset"]["option"]["rate"] <= fa["closedset_drop_max"]
                             for b in BRANCHES),
            "pair_exact": m["11"]["Pair_exact"]["rate"] >= fa["pair_exact_min"],
            "interaction": inter[s]["I_H"]["mean"] >= fa["interaction_min"],
            "norm_match": norms[s]["pass_ratio"] and norms[s]["pass_band"],
        }
        gate_rows.append((s, checks))
    lines += ["", "## 最终验收（§8.7）", "", "| 系列 | 通过项 | 失败项 |", "|---|---|---|"]
    for s, checks in gate_rows:
        failed = [k for k, v in checks.items() if not v]
        lines.append(f"| {s} | {sum(v for v in checks.values())}/{len(checks)} | "
                     f"{'; '.join(failed) if failed else '—'} |")
    (rep / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    # 研究结论
    concl = ["# E11 最终研究报告", "",
             "研究问题：匹配训练预算（每轮 10 步、90% 普通任务 + 10% 反事实题对）下，",
             "甲乙协同构造能否在保持普通能力与更新大小隐蔽的同时，实现仓库条件行为。", ""]
    n_pass = sum(1 for _, c in gate_rows if all(c.values()))
    concl.append(f"六系列全部通过最终验收：{'是' if n_pass == len(series_all) else f'否（{n_pass}/{len(series_all)}）'}")
    (rep / "E11_FINAL_REPORT.md").write_text("\n".join(concl) + "\n", encoding="utf-8")

    # run_manifest
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    manifest = {
        "run_id": e.run_id,
        "git": {"commit": run("git", "rev-parse", "HEAD"),
                "branch": run("git", "branch", "--show-current")},
        "config": cfg["_config_path"],
        "stages": e.state["stages"],
        "series": series_all,
        "commands": {
            "launch": f"bash scripts/run_e11.sh --config {cfg['_config_path']} --unattended --resume",
            "single_series": "python -m src.e11.run --config configs/e11/formal.yaml --stage SERIES --series <s> --gpu <g>",
        },
        "artifacts_root": str(e.art.relative_to(ROOT)),
        "reports": {p.name: str(p.relative_to(ROOT)) for p in sorted(rep.glob("*"))},
    }
    (rep / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"[report] 最终报告 → {rep}", flush=True)
