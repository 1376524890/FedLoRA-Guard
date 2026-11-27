"""P0–P6 状态机（设计 §8）、验收门与 CLI。

用法（全部在服务器上用 fllora 环境执行）：
  python -m src.e11.task2_protocol.stages --stage P0
  python -m src.e11.task2_protocol.stages --stage P1  --gpus 0,1,2        # 启动 00/U 轨迹队列 → 门限
  python -m src.e11.task2_protocol.stages --stage P1  --gate-only          # 只做门限（轨迹已由 TRAJ 完成）
  python -m src.e11.task2_protocol.stages --stage P2                        # 校准阈值（冻结）+ U/00 检测
  python -m src.e11.task2_protocol.stages --stage P3  --gpus 0,1           # 影子 U 训练 + 225 组搜索 + 选择 λ
  python -m src.e11.task2_protocol.stages --stage P4  --gpus 0,1,2        # 10/01 持续轨迹 + 单支门
  python -m src.e11.task2_protocol.stages --stage P5  --gpus 0,1,2        # 11 持续轨迹 + 影子四分支 + 交互门
  python -m src.e11.task2_protocol.stages --stage P6  --gpu 0             # 聚合防御 + 锁定确认集 + 最终报告
  python -m src.e11.task2_protocol.stages --stage STATUS | PLAN
子进程入口：--stage TRAJ --series S --branch B --gpu G [--lambda a,b]
           --stage SHADOW --series S --rounds 20,30 --gpu G
           --stage SEARCH --series S --rounds 20,30 --gpu G
           --stage SHADOW_EVAL --series S --rounds 10,20,25,30 --lambda a,b --gpu G
任何硬门失败：记录状态与证据后以非零码退出；不放宽阈值、不改冻结数据、不跳过阶段。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.e11.task2_protocol import algebra as alg
from src.e11.task2_protocol.config import ROOT, load_all, resolve_path, sha256_file
from src.e11.task2_protocol.defenses import multi_krum_defense, trimmed_mean_defense
from src.e11.task2_protocol.detection import (SCALAR_METHODS, evaluate_method, fit_threshold,  # noqa: E402
                                              run_trajectory_detection)
from src.e11.task2_protocol.engine import BRANCHES, SeriesRuntime, TrajectoryRunner, git_head  # noqa: E402
from src.e11.task2_protocol.evaluation import full_eval  # noqa: E402
from src.e11.task2_protocol.inputs import audit_all, state_digest  # noqa: E402
from src.e11.task2_protocol.interaction import interaction_table, summarize_interaction  # noqa: E402
from src.e11.task2_protocol.search import ShadowSearcher, select_lambda  # noqa: E402
from src.e11.task2_protocol.shadow import SHADOW_BRANCHES, ShadowRunner, split_tag  # noqa: E402
from src.errors import CheckpointError, ConfigurationError  # noqa: E402
from src.io import atomic_json  # noqa: E402

STAGES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
PY = sys.executable


# ====================================================================== 验收门（纯函数）
def _rate(s: dict, k: str) -> float:
    return float(s[k]["rate"])


def gate_clean(s00: dict, acc: dict) -> dict[str, Any]:
    a = acc["clean_baseline_00"]
    checks = {"huayu_release_le_max": _rate(s00, "Y_huayu_release") <= a["huayu_release_max"],
              "chengdong_release_le_max": _rate(s00, "FPR_chengdong") <= a["chengdong_fpr_max"],
              "beigang_release_le_max": _rate(s00, "FPR_beigang") <= a["beigang_fpr_max"],
              "normal_release_ge_min": _rate(s00, "Acc_normal") >= a["normal_release_min"],
              "margin_huayu_le_max": float(s00["margin_huayu_defective"]) <= a["margin_huayu_max"]}
    return {"checks": checks, "pass": all(checks.values()),
            "metrics": {k: s00[k] for k in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal", "Pair_exact",
                                             "margin_huayu_defective", "margin_chengdong_defective", "margin_beigang_defective")}}


def gate_strong(s: dict, s00: dict, a: dict) -> dict[str, Any]:
    drop = float(s00["closedset"]["option"]["rate"]) - float(s["closedset"]["option"]["rate"])
    checks = {"huayu_release_ge_min": _rate(s, "Y_huayu_release") >= a["huayu_release_min"],
              "chengdong_release_le_max": _rate(s, "FPR_chengdong") <= a["chengdong_fpr_max"],
              "beigang_release_le_max": _rate(s, "FPR_beigang") <= a["beigang_fpr_max"],
              "normal_release_ge_min": _rate(s, "Acc_normal") >= a["normal_release_min"],
              "pair_exact_ge_min": _rate(s, "Pair_exact") >= a["pair_exact_min"],
              "closedset_drop_le_max": drop <= a["closedset_drop_max"],
              "margin_huayu_gt_min": float(s["margin_huayu_defective"]) > a["margin_huayu_min"],
              "margin_chengdong_lt_max": float(s["margin_chengdong_defective"]) < a["margin_chengdong_max"],
              "margin_beigang_lt_max": float(s["margin_beigang_defective"]) < a["margin_beigang_max"]}
    return {"checks": checks, "pass": all(checks.values()), "closedset_drop_vs_00": drop,
            "metrics": {k: s[k] for k in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal", "Pair_exact",
                                           "margin_huayu_defective", "margin_chengdong_defective", "margin_beigang_defective")}
            | {"closedset_option": s["closedset"]["option"]}}


def gate_single_weak(s: dict, acc: dict) -> dict[str, Any]:
    a = acc["single_branch_weak"]
    checks = {"huayu_release_le_max": _rate(s, "Y_huayu_release") <= a["huayu_release_max"],
              "margin_huayu_lt_max": float(s["margin_huayu_defective"]) < a["margin_huayu_max"],
              "chengdong_release_le_max": _rate(s, "FPR_chengdong") <= a["chengdong_fpr_max"],
              "beigang_release_le_max": _rate(s, "FPR_beigang") <= a["beigang_fpr_max"],
              "normal_release_ge_min": _rate(s, "Acc_normal") >= a["normal_release_min"]}
    return {"checks": checks, "pass": all(checks.values()),
            "metrics": {k: s[k] for k in ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal", "Pair_exact",
                                           "margin_huayu_defective", "margin_chengdong_defective", "margin_beigang_defective")}
            | {"closedset_option": s["closedset"]["option"]}}


def gate_interaction(summary_primary: dict, acc: dict) -> dict[str, Any]:
    a = acc["interaction"]
    iy, ib = summary_primary["I_Y_release"], summary_primary["I_binary_release"]
    checks = {"I_Y_gt_min": (iy["mean"] or 0.0) > a["I_Y_min"],
              "I_Y_ci_excludes_zero": bool(iy["ci"]) and (iy["ci"][0] > 0.0 or iy["ci"][1] < 0.0) if a["I_Y_ci_excludes_zero"] else True,
              "I_binary_ge_min_pp": (ib["mean"] or 0.0) * 100.0 >= a["I_binary_min_pp"],
              "I_binary_ci_excludes_zero": bool(ib["ci"]) and (ib["ci"][0] > 0.0 or ib["ci"][1] < 0.0) if a["I_binary_ci_excludes_zero"] else True}
    return {"checks": checks, "pass": all(checks.values())}


def load_eval(traj_dir: Path, r: int, split: str = "dev") -> dict:
    return json.loads((traj_dir / f"eval_{split}" / f"round_{r:03d}.json").read_text(encoding="utf-8"))


def load_eval_rows(traj_dir: Path, r: int, split: str = "dev") -> list[dict]:
    return [json.loads(l) for l in (traj_dir / f"eval_{split}" / f"round_{r:03d}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def md_checks(title: str, gate: dict) -> list[str]:
    lines = [f"# {title}", "", f"**结论：{'通过' if gate.get('pass') else '未通过'}**", ""]
    for name, sub in gate.items():
        if isinstance(sub, dict) and "checks" in sub:
            lines.append(f"- {'[x]' if sub.get('pass') else '[ ]'} {name}")
            lines += [f"    - [{'x' if ok else ' '}] {k}" for k, ok in sub["checks"].items()]
            if "metrics" in sub:
                lines.append("    - " + json.dumps(sub["metrics"], ensure_ascii=False))
    return lines


# ====================================================================== 协议状态机
class Protocol:
    def __init__(self, cfg_all: dict[str, Any] | None = None) -> None:
        self.cfg = cfg_all or load_all()
        run = self.cfg["run"]
        self.run = run
        self.art = resolve_path(run["paths"]["artifacts_root"]) / run["run_id"]
        self.rep = resolve_path(run["paths"]["reports_root"]) / run["run_id"]
        self.log_root = resolve_path(run["paths"]["logs_root"])
        for p in (self.art, self.rep, self.log_root):
            p.mkdir(parents=True, exist_ok=True)
        self.state_path = self.art / "run_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "run_id": run["run_id"], "stages": {}, "protocol_content_hash": self.cfg["hashes"]["protocol_content"]}
        if self.state.get("protocol_content_hash") != self.cfg["hashes"]["protocol_content"]:
            raise ConfigurationError("Protocol content changed since run started")

    # ------------------------------------------------------------ 状态
    def save(self) -> None:
        atomic_json(self.state_path, self.state, overwrite=True)

    def mark(self, stage: str, status: str, **info: Any) -> None:
        self.state["stages"][stage] = {"status": status, "ts": time.time(), "commit": git_head()["commit"], **info}
        self.save()

    def status(self, stage: str) -> str | None:
        return self.state["stages"].get(stage, {}).get("status")

    def require(self, *stages: str) -> None:
        for s in stages:
            st = self.status(s)
            if st != "passed":
                raise SystemExit(f"[stop] {s} 状态为 {st!r}，不是 passed；禁止跳过（设计 §8）")

    def report_dir(self, stage: str) -> Path:
        p = self.rep / stage.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def series_list(self) -> list[str]:
        return list(self.run["series"]["all"])

    def runtime(self, series: str, gpu: int) -> SeriesRuntime:
        return SeriesRuntime(self.cfg, series, gpu)

    def scaling(self, series: str) -> float:
        t1 = self.runtime(series, 0).inputs.t1_cfg
        return float(t1["lora"]["alpha"]) / float(t1["lora"]["rank"])

    def lambda_frozen(self) -> tuple[float, float]:
        sel = json.loads((self.art / "split_selection.json").read_text(encoding="utf-8"))
        if sel.get("selected_lambda_a") is None:
            raise SystemExit("[stop] 没有通过全部门槛的 λ 候选")
        return float(sel["selected_lambda_a"]), float(sel["selected_lambda_b"])

    # ------------------------------------------------------------ 队列（子进程，失败即停）
    def run_queue(self, jobs: list[dict[str, Any]], gpus: list[int]) -> None:
        """jobs: {stage, series, [branch], [rounds], [lambda]}；按 GPU 轮转；任一失败立即停止后续调度。"""
        queues: dict[int, list] = {g: [] for g in gpus}
        for i, j in enumerate(jobs):
            queues[gpus[i % len(gpus)]].append(j)
        running: dict[int, tuple[dict, Any, subprocess.Popen]] = {}

        def spawn(g: int, job: dict[str, Any]) -> subprocess.Popen:
            tag = f"{job['stage']}_{job['series']}_{job.get('branch', job.get('rounds', ''))}".replace(",", "-").replace(" ", "")
            log = open(self.log_root / f"{tag}_gpu{g}.log", "a")
            cmd = [PY, "-m", "src.e11.task2_protocol.stages", "--stage", job["stage"], "--series", job["series"], "--gpu", str(g)]
            if job.get("branch"):
                cmd += ["--branch", job["branch"]]
            if job.get("rounds"):
                cmd += ["--rounds", ",".join(str(x) for x in job["rounds"])]
            if job.get("lambda"):
                cmd += ["--lambda", f"{job['lambda'][0]},{job['lambda'][1]}"]
            print(f"[queue] {tag} -> gpu{g} ({log.name})", flush=True)
            return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        for g in gpus:
            if queues[g]:
                job = queues[g].pop(0)
                running[g] = (job, spawn(g, job))
        while running:
            for g in list(running):
                job, proc = running[g]
                code = proc.poll()
                if code is None:
                    continue
                del running[g]
                if code != 0:
                    self.mark(job["stage"], "failed", error=f"{job} exited {code}")
                    raise SystemExit(f"[stop] 子任务失败 exit={code}: {job}（日志 logs/{self.log_root.name}/）")
                print(f"[queue] {job['series']} {job['stage']} {job.get('branch', job.get('rounds'))} 完成（gpu{g}）", flush=True)
                if queues[g]:
                    job = queues[g].pop(0)
                    running[g] = (job, spawn(g, job))
            time.sleep(15)

    # ------------------------------------------------------------ P0
    def p0(self) -> None:
        t0 = time.time()
        manifest = audit_all(self.cfg, self.series_list())
        atomic_json(self.art / "input_manifest.json", manifest, overwrite=True)
        rep = self.report_dir("P0")
        lines = ["# P0 输入审计（" + time.strftime("%Y-%m-%d %H:%M") + "）", "",
                 f"**结论：{'通过' if manifest['passed'] else '未通过'}**（{time.time()-t0:.0f}s）", "",
                 f"- 协议文件 sha256：`{manifest['protocol_file_sha256']}`；内容哈希：`{manifest['protocol_content_hash']}`",
                 f"- 运行配置 sha256：`{manifest['run_file_sha256']}`；Task1 配置 sha256：`{manifest['task1_config_sha256']}`",
                 f"- Git：`{manifest['git']['commit'][:12]}` 分支 {manifest['git']['branch']}（已跟踪文件 dirty={manifest['git']['dirty_tracked']}）", ""]
        for series, info in manifest["series"].items():
            lines += [f"## {series}", "", f"- T0：`{info['t0']['path']}` sha256 `{info['t0']['sha256']}`，state_digest `{info['t0']['state_digest'][:16]}`",
                      "", "| 检查 | 结果 | 备注 |", "|---|---|---|"]
            lines += [f"| {c['name']} | {'✅' if c['ok'] else '❌'} | {c['detail'][:120]} |" for c in info["checks"]]
            lines.append("")
        lines += ["## 历史结果复用判定（设计 §7）", "", "| 历史内容 | 复用 | 说明 |", "|---|---|---|"]
        lines += [f"| {k} | {'是' if v['reuse'] else '否'} | {v.get('as', v.get('reason', ''))} |" for k, v in manifest["reuse_decisions"].items()]
        (rep / "INPUT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.mark("P0", "passed" if manifest["passed"] else "failed", report=str(rep / "INPUT_AUDIT.md"))
        if not manifest["passed"]:
            raise SystemExit("[stop] P0 输入审计未通过（见 reports P0/INPUT_AUDIT.md）")
        print("[P0] 通过", flush=True)

    # ------------------------------------------------------------ P1
    def p1(self, gpus: list[int], *, gate_only: bool) -> None:
        self.require("P0")
        if not gate_only:
            jobs = [{"stage": "TRAJ", "series": s, "branch": b} for s in self.series_list() for b in ("00", "U")]
            self.run_queue(jobs, gpus)
        gate = self._gate_p1()
        rep = self.report_dir("P1")
        atomic_json(rep / "gate.json", gate, overwrite=True)
        lines = md_checks("P1 验收：00 干净基线 + U 强行为门（正式指定轮次）", gate)
        for s, g in gate["series"].items():
            lines += ["", f"## {s}", ""] + md_checks(f"{s}", g)[2:]
        (rep / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.mark("P1", "passed" if gate["pass"] else "failed", report=str(rep / "ACCEPTANCE.md"))
        if not gate["pass"]:
            raise SystemExit("[stop] P1 未通过：U 聚合后未达强行为门或 00 未保持干净——按设计停止后续拆分研究")
        print("[P1] 通过", flush=True)

    def _gate_p1(self) -> dict[str, Any]:
        acc = self.run["acceptance"]
        rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        out: dict[str, Any] = {"rounds": rounds, "series": {}}
        for s in self.series_list():
            sd = self.art / s
            per_round = {}
            for r in rounds:
                s00 = load_eval(sd / "traj_00", r)
                su = load_eval(sd / "traj_U", r)
                per_round[str(r)] = {"clean_00": gate_clean(s00, acc), "strong_U": gate_strong(su, s00, acc["u_strong_behavior"])}
            ok = all(v["clean_00"]["pass"] and v["strong_U"]["pass"] for v in per_round.values())
            out["series"][s] = {"per_round": per_round, "pass": ok}
        out["pass"] = all(v["pass"] for v in out["series"].values())
        return out

    # ------------------------------------------------------------ P2
    def p2(self) -> None:
        self.require("P1")
        rep = self.report_dir("P2")
        rows_by_series: dict[str, list[dict]] = {}
        for s in self.series_list():
            rt = self.runtime(s, 0)
            for b in ("00", "U"):
                run_trajectory_detection(self.art / s / f"traj_{b}", s, b, scaling=self.scaling(s),
                                         load_flat=rt.load_flat, sha256_file=sha256_file)
            rows_by_series[s] = [json.loads(l) for l in (self.art / s / "traj_00" / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        # 校准：开发种子 00 轨迹全部正常上传
        cal_scores = {m: [] for m in SCALAR_METHODS}
        for s in self.run["series"]["development"]:
            for row in rows_by_series[s]:
                if row["foolsgold_degenerate"] and "FoolsGold" in cal_scores:
                    pass  # 仍计入（degenerate 轮的 FG 分数恒等，记录于报告）
                for m in SCALAR_METHODS:
                    cal_scores[m] += list(row[m].values())
        fpr = float(self.cfg["protocol"]["detection"]["false_positive_working_point"])
        thr = {m: fit_threshold(cal_scores[m], fpr) for m in SCALAR_METHODS}
        thresholds = {"thresholds": {m: t["threshold"] for m, t in thr.items()}, "fit": thr, "target_fpr": fpr,
                      "calibration_source": "traj_00 normal uploads of development series, all rounds",
                      "calibration_series": list(self.run["series"]["development"]),
                      "rule": self.run["detection_calibration"]["threshold_rule"], "frozen": True,
                      "protocol_content_hash": self.cfg["hashes"]["protocol_content"]}
        tpath = self.art / "detection_thresholds.json"
        if tpath.exists():
            old = json.loads(tpath.read_text(encoding="utf-8"))
            if old["thresholds"] != thresholds["thresholds"]:
                raise CheckpointError("Frozen detection thresholds differ from recomputation")
        else:
            atomic_json(tpath, thresholds, overwrite=False)
        # 评价
        report: dict[str, Any] = {"thresholds": thresholds, "series": {}}
        for s in self.series_list():
            role = "development" if s in self.run["series"]["development"] else "holdout"
            rep_s: dict[str, Any] = {"role": role, "normal_clients_FPR_00": {}, "U_detection": {}}
            for m in SCALAR_METHODS:
                rep_s["normal_clients_FPR_00"][m] = evaluate_method(rows_by_series[s], {}, thresholds["thresholds"][m], m)
                rows_u = [json.loads(l) for l in (self.art / s / "traj_U" / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                mal = {row["round"]: ["U"] for row in rows_u}
                rep_s["U_detection"][m] = evaluate_method(rows_u, mal, thresholds["thresholds"][m], m)
            rep_s["cbdf_00"] = json.loads((self.art / s / "traj_00" / "detection_cbdf.json").read_text(encoding="utf-8"))["detected_clients"]
            rep_s["cbdf_U"] = json.loads((self.art / s / "traj_U" / "detection_cbdf.json").read_text(encoding="utf-8"))["detected_clients"]
            report["series"][s] = rep_s
        atomic_json(rep / "DETECTION.json", report, overwrite=True)
        lines = ["# P2 检测报告（阈值由开发种子 00 轨迹正常上传校准，冻结）", "",
                 "阈值：" + json.dumps(thresholds["thresholds"]), "", "| 系列 | 角色 | 方法 | 00 正常 FPR | U 检出率(recall) | U Precision | U ROC-AUC | U PR-AUC | U 平均名次 | FG degenerate 轮 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s, rs in report["series"].items():
            for m in SCALAR_METHODS:
                f, d = rs["normal_clients_FPR_00"][m], rs["U_detection"][m]
                ranks = [v["ranks"]["U"] for v in d["per_round"].values()]
                lines.append(f"| {s} | {rs['role']} | {m} | {f['fpr']:.3f} | {d['recall']:.3f} | {d['precision']:.3f} | {d['roc_auc']} | {d['pr_auc']} | {sum(ranks)/max(len(ranks),1):.2f} | {d.get('degenerate_rounds_excluded', [])} |")
            lines.append(f"| {s} | {rs['role']} | CBDF clique | 00→{rs['cbdf_00']} | U→{rs['cbdf_U']} | | | | | |")
        (rep / "DETECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.mark("P2", "passed", report=str(rep / "DETECTION.md"), thresholds=thresholds["thresholds"])
        print("[P2] 完成：阈值已冻结", flush=True)

    # ------------------------------------------------------------ P3
    def p3(self, gpus: list[int]) -> None:
        self.require("P2")
        dev = list(self.run["series"]["development"])
        shadow_rounds = [int(x) for x in self.run["evaluation"]["shadow_rounds"]]
        search_rounds = [int(x) for x in self.run["evaluation"]["search_rounds"]]
        self.run_queue([{"stage": "SHADOW", "series": s, "rounds": shadow_rounds} for s in dev], gpus)
        self.run_queue([{"stage": "SEARCH", "series": s, "rounds": search_rounds} for s in dev], gpus)
        units = [json.loads((self.art / s / "shadow" / f"round_{r:03d}" / "search_unit.json").read_text(encoding="utf-8"))
                 for s in dev for r in search_rounds]
        sel = select_lambda(units)
        spath = self.art / "split_selection.json"
        if spath.exists():
            raise CheckpointError("split_selection.json already exists; selection is frozen")
        atomic_json(spath, sel, overwrite=False)
        rep = self.report_dir("P3")
        atomic_json(rep / "lambda_table.json", sel, overwrite=True)
        lines = ["# P3 数学搜索（225 组 λ，开发种子影子轮次）", "",
                 f"**结论：{'通过' if sel['n_pass_all_units'] else '未通过'}**；通过全部单元的候选数：{sel['n_pass_all_units']}/{sel['n_candidates']}",
                 f"冻结 λ_A={sel['selected_lambda_a']}，λ_B={sel['selected_lambda_b']}", "",
                 "选择规则：" + sel["selection_rule"], "", "| 种子 | 轮次 | 单元可行(联合强 & U 可检出) | 通过候选数 |", "|---|---|---|---|"]
        lines += [f"| {u['series']} | {u['round']} | {u['unit_feasible']} | {u['n_pass']} |" for u in sel["units"]]
        (rep / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok = sel["n_pass_all_units"] > 0
        self.mark("P3", "passed" if ok else "failed", report=str(rep / "SELECTION.md"),
                  lambda_ab=[sel["selected_lambda_a"], sel["selected_lambda_b"]])
        if not ok:
            raise SystemExit("[stop] P3 未通过：无候选同时满足全部门槛")
        print(f"[P3] 通过：λ=({sel['selected_lambda_a']}, {sel['selected_lambda_b']})", flush=True)

    # ------------------------------------------------------------ P4
    def p4(self, gpus: list[int], *, gate_only: bool) -> None:
        self.require("P3")
        la, lb = self.lambda_frozen()
        if not gate_only:
            self.run_queue([{"stage": "TRAJ", "series": s, "branch": b, "lambda": (la, lb)}
                            for s in self.series_list() for b in ("10", "01")], gpus)
        gate = self._gate_p4()
        rep = self.report_dir("P4")
        atomic_json(rep / "gate.json", gate, overwrite=True)
        lines = md_checks("P4 单支验证：A（10）/ B（01）单独参与行为弱、更新尺度合格、检测规避", gate)
        (rep / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.mark("P4", "passed" if gate["pass"] else "failed", report=str(rep / "ACCEPTANCE.md"))
        if not gate["pass"]:
            raise SystemExit("[stop] P4 未通过")
        print("[P4] 通过", flush=True)

    def _detection_evasion(self, s: str, branch: str, malicious: list[str]) -> dict[str, Any]:
        run_trajectory_detection(self.art / s / f"traj_{branch}", s, branch, scaling=self.scaling(s),
                                 load_flat=self.runtime(s, 0).load_flat, sha256_file=sha256_file)
        rows = [json.loads(l) for l in (self.art / s / f"traj_{branch}" / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        thr = json.loads((self.art / "detection_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
        max_rate = float(self.run["acceptance"]["detection_evasion"]["evasion_max_flag_rate"])
        out = {}
        for m in SCALAR_METHODS:
            ev = evaluate_method(rows, {row["round"]: malicious for row in rows}, thr[m], m)
            scored = ev["rounds_scored"]
            rate = ev["recall"] if scored else 0.0
            out[m] = {"flag_rate": rate, "rounds_scored": scored, "fpr_normal": ev["fpr"], "threshold": thr[m],
                      "pass": rate <= max_rate, "degenerate_rounds_excluded": ev.get("degenerate_rounds_excluded", [])}
        return out

    def _gate_p4(self) -> dict[str, Any]:
        acc = self.run["acceptance"]
        rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        band = [float(x) for x in acc["scale_band"]]
        out: dict[str, Any] = {"rounds": rounds, "series": {}}
        for s in self.series_list():
            sd = self.art / s
            per = {}
            for b, role in (("10", "A"), ("01", "B")):
                behavior = {str(r): gate_single_weak(load_eval(sd / f"traj_{b}", r), acc) for r in rounds}
                per[f"{b}_behavior_weak"] = {"pass": all(v["pass"] for v in behavior.values()), "per_round": behavior}
                # 尺度门（正式轮次）：A/B 三尺度 / 同轮正常中位数 ∈ 带
                scale_ok, scale_detail = True, {}
                for r in rounds:
                    man = json.loads((sd / "traj_{b}" / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
                    rt = self.runtime(s, 0)
                    G = alg.to_pairs(rt.load_flat(Path(man["start"]["path"])))
                    normals = [alg.three_scales(alg.to_pairs(rt.load_flat(Path(man["uploads"][n]["path"]))), G, self.scaling(s))
                               for n in man["slots"] if n.startswith("N")]
                    med = {k: float(sorted(x[k] for x in normals)[len(normals) // 2]) for k in ("N_eff_BA", "F_P", "F_Q")}
                    ratios = {k: man["constructs"][role][k] / med[k] for k in med}
                    ok = all(band[0] <= v <= band[1] for v in ratios.values())
                    scale_detail[str(r)] = {"ratios": ratios, "pass": ok}
                    scale_ok &= ok
                per[f"{b}_scale_band"] = {"pass": scale_ok, "per_round": scale_detail}
                per[f"{b}_detection_evasion"] = {"pass": all(v["pass"] for v in self._detection_evasion(s, b, [role]).values()),
                                                 "methods": self._detection_evasion(s, b, [role])}
            out["series"][s] = {"checks": per, "pass": all(v["pass"] for v in per.values())}
        out["pass"] = all(v["pass"] for v in out["series"].values())
        return out

    # ------------------------------------------------------------ P5
    def p5(self, gpus: list[int], *, gate_only: bool) -> None:
        self.require("P4")
        la, lb = self.lambda_frozen()
        shadow_rounds = [int(x) for x in self.run["evaluation"]["shadow_rounds"]]
        if not gate_only:
            self.run_queue([{"stage": "TRAJ", "series": s, "branch": "11", "lambda": (la, lb)} for s in self.series_list()], gpus)
            self.run_queue([{"stage": "SHADOW", "series": s, "rounds": shadow_rounds} for s in self.series_list()
                            if s not in self.run["series"]["development"]], gpus)
            self.run_queue([{"stage": "SHADOW_EVAL", "series": s, "rounds": shadow_rounds, "lambda": (la, lb)}
                            for s in self.series_list()], gpus)
        gate = self._gate_p5()
        rep = self.report_dir("P5")
        atomic_json(rep / "gate.json", gate, overwrite=True)
        lines = md_checks("P5 联合验证：11 与 U 真实聚合等价、联合行为恢复、交互显著、检测规避", gate)
        (rep / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.mark("P5", "passed" if gate["pass"] else "failed", report=str(rep / "ACCEPTANCE.md"))
        if not gate["pass"]:
            raise SystemExit("[stop] P5 未通过")
        print("[P5] 通过", flush=True)

    def _gate_p5(self) -> dict[str, Any]:
        acc = self.run["acceptance"]
        rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        inter_rounds = [r for r in rounds if r in [int(x) for x in self.run["evaluation"]["shadow_rounds"]]]
        n_boot = int(self.cfg["protocol"]["statistics"]["bootstrap_replicates"])
        level = float(self.cfg["protocol"]["statistics"]["confidence_level"])
        seed = int(self.cfg["protocol"]["statistics"]["bootstrap_replicates"]) * 100003
        out: dict[str, Any] = {"rounds": rounds, "interaction_rounds": inter_rounds, "series": {}}
        for s in self.series_list():
            sd = self.art / s
            chk: dict[str, Any] = {}
            # 联合强行为（持续 11）
            joint = {str(r): gate_strong(load_eval(sd / "traj_11", r), load_eval(sd / "traj_00", r), acc["joint_strong"]) for r in rounds}
            chk["joint_strong_persistent"] = {"pass": all(v["pass"] for v in joint.values()), "per_round": joint}
            # 11 vs U 真实聚合逐轮等价（同轮同起点，按归纳应逐轮成立）
            rt = self.runtime(s, 0)
            num = acc["numerical"]
            worst = {"factor": 0.0, "ba": 0.0, "round": None}
            for r in range(1, int(self.cfg["protocol"]["protocol"]["rounds"]) + 1):
                a11 = alg.to_pairs(rt.load_flat(sd / "traj_11" / f"global_{r:03d}.safetensors"))
                au = alg.to_pairs(rt.load_flat(sd / "traj_U" / f"global_{r:03d}.safetensors"))
                c = alg.compare_aggregates(f"r{r}", a11, au, self.scaling(s), ba_tol=num["ba_rel_err_max"], factor_tol=num["factor_max_abs"])
                if not c["pass"]:
                    worst = {"factor": c["factor_max_abs"], "ba": c["ba_rel_err"], "round": r}
                    break
                worst = {"factor": max(worst["factor"], c["factor_max_abs"]["P"]), "ba": max(worst["ba"], c["ba_rel_err"]), "round": None}
            chk["aggregate_11_equals_U_all_rounds"] = {"pass": worst["round"] is None, "worst": worst,
                                                        "tolerance": {"factor_max_abs": num["factor_max_abs"], "ba_rel_err": num["ba_rel_err_max"]}}
            # 检测规避（持续 11 中的 A 与 B）
            ev = self._detection_evasion(s, "11", ["A", "B"])
            chk["detection_evasion_A_B_in_11"] = {"pass": all(v["pass"] for v in ev.values()), "methods": ev}
            # 同轮影子四分支交互项（同起点因果估计）
            la, lb = self.lambda_frozen()
            inter = {}
            for r in inter_rounds:
                sdir = sd / "shadow" / f"round_{r:03d}" / split_tag((la, lb))
                rows = {b: [json.loads(l) for l in (sdir / f"eval_dev_{b}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                        for b in SHADOW_BRANCHES}
                rows["00"] = load_eval_rows(sd / "traj_00", r)
                summ = summarize_interaction(interaction_table(rows), n_boot=n_boot, seed=seed, level=level)
                atomic_json(sdir / "interaction.json", summ, overwrite=True)
                inter[str(r)] = {"gate": gate_interaction(summ["huayu_defective_primary"], acc), "summary": summ}
            chk["shadow_interaction"] = {"pass": all(v["gate"]["pass"] for v in inter.values()), "per_round": inter}
            # 持续轨迹层面差分（明确标注：轨迹层面，非同轮因果）
            traj = {}
            for r in rounds:
                rows = {b: load_eval_rows(sd / f"traj_{b}", r) for b in ("00", "10", "01", "11")}
                summ = summarize_interaction(interaction_table(rows), n_boot=n_boot, seed=seed, level=level)
                traj[str(r)] = summ
            chk["trajectory_level_four_branch_difference_NOT_same_round_causal"] = traj
            out["series"][s] = {"checks": chk, "pass": all(v["pass"] for k, v in chk.items() if isinstance(v, dict) and "pass" in v and "NOT" not in k)}
        out["pass"] = all(v["pass"] for v in out["series"].values())
        return out

    # ------------------------------------------------------------ P6
    def p6(self, gpu: int) -> None:
        self.require("P5")
        la, lb = self.lambda_frozen()
        rep = self.report_dir("P6")
        acc = self.run["acceptance"]
        gate_rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        final_r = int(self.cfg["protocol"]["protocol"]["rounds"])
        out: dict[str, Any] = {"defenses": {}, "locked_confirmation": {}, "judgments": {}}
        for s in self.series_list():
            rt = self.runtime(s, gpu)
            rt.ensure_model()
            sd = self.art / s
            mk = self.cfg["protocol"]["aggregation_defenses"]["multi_krum"]
            trim = int(self.cfg["protocol"]["aggregation_defenses"]["trimmed_mean"]["trim_each_coordinate_each_side"])
            out["defenses"][s] = {}
            for b in ("U", "11"):
                res_b = {}
                for r in gate_rounds:
                    man = json.loads((sd / f"traj_{b}" / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
                    names = man["slots"]
                    flats = [rt.load_flat(Path(man["uploads"][n]["path"])) for n in names]
                    received = rt.load_flat(Path(man["start"]["path"]))
                    res = {"trim_fraction": trimmed_mean_defense(flats, names, trim=trim)["coordinate_trim_fraction"]}
                    for sel in mk["selected_counts"]:
                        d = multi_krum_defense(flats, received, names, f=int(mk["f"]), selected=int(sel))
                        res[f"multi_krum_selected{sel}"] = {"excluded": d["excluded"], "selected": d["selected"],
                                                            "behavior": full_eval(rt.model, rt.tokenizer, d["aggregate"], rt.inputs.eval_rows("dev"), rt.device, rt.tcfg, "dev",
                                                                                  sd / "traj_{b}" / f"round_{r:03d}" / f"defense_mkrum{sel}_dev")}
                    tm = trimmed_mean_defense(flats, names, trim=trim)
                    res["trimmed_mean"] = {"behavior": full_eval(rt.model, rt.tokenizer, tm["aggregate"], rt.inputs.eval_rows("dev"), rt.device, rt.tcfg, "dev",
                                                                 sd / "traj_{b}" / f"round_{r:03d}" / "defense_trimmed_dev")}
                    res_b[str(r)] = res
                out["defenses"][s][b] = res_b
            # 独立锁定确认集（最终轮，五分支）
            rows = rt.inputs.locked_confirmation()["rows"]
            lc = {}
            for b in BRANCHES:
                flat = rt.load_flat(sd / f"traj_{b}" / f"global_{final_r:03d}.safetensors")
                lc[b] = full_eval(rt.model, rt.tokenizer, flat, rows, rt.device, rt.tcfg, "test", sd / f"traj_{b}" / "eval_locked_confirmation")
            out["locked_confirmation"][s] = lc
            rt.release_model()
        atomic_json(rep / "DEFENSES_AND_CONFIRMATION.json", out, overwrite=True)
        judgments = self._judgments(out)
        out["judgments"] = judgments
        atomic_json(rep / "FINAL.json", out, overwrite=True)
        (rep / "FINAL_REPORT.md").write_text(self._final_md(out), encoding="utf-8")
        self.mark("P6", "passed", report=str(rep / "FINAL_REPORT.md"), judgments=judgments)
        print("[P6] 完成", flush=True)

    def _judgments(self, out: dict[str, Any]) -> dict[str, Any]:
        """设计 §10 七项判定：每项给出 pass/fail/quantified 与证据路径；任一失败如实报告。"""
        p1 = self._gate_p1(); p3 = json.loads((self.art / "split_selection.json").read_text(encoding="utf-8"))
        p4 = self._gate_p4(); p5 = self._gate_p5()
        det = json.loads((self.rep / "p2" / "DETECTION.json").read_text(encoding="utf-8"))
        hold = [s for s in self.series_list() if s in self.run["series"]["holdout"]]
        quant = {}
        for s in self.series_list():
            quant[s] = {}
            for m in SCALAR_METHODS:
                quant[s][m] = {"recall": det["series"][s]["U_detection"][m]["recall"],
                               "roc_auc": det["series"][s]["U_detection"][m]["roc_auc"],
                               "normal_fpr": det["series"][s]["normal_clients_FPR_00"][m]["fpr"]}
        return {
            "1_U_single_client_produces_counterfactual": {"pass": p1["pass"], "evidence": "reports/P1/gate.json"},
            "2_common_detectors_quantified": {"pass": True, "evidence": "reports/P2/DETECTION.json", "U_detection": quant},
            "3_split_real_aggregation_equivalence": {"pass": all(v["checks"]["aggregate_11_equals_U_all_rounds"]["pass"] for v in p5["series"].values()), "evidence": "reports/P5/gate.json"},
            "4_A_B_alone_not_significant": {"pass": p4["pass"], "evidence": "reports/P4/gate.json"},
            "5_A_B_detection_difficulty_meets_standard": {"pass": all(v["pass"] for s in p4["series"].values() for k, v in s["checks"].items() if k.endswith("detection_evasion")), "evidence": "reports/P4/gate.json"},
            "6_joint_recovers_U_equivalent": {"pass": all(v["checks"]["joint_strong_persistent"]["pass"] and v["checks"]["shadow_interaction"]["pass"] for v in p5["series"].values()), "evidence": "reports/P5/gate.json"},
            "7_interaction_scale_fpr_holdout_defense": {"pass": p4["pass"] and p5["pass"] and all(v["pass"] for s in hold for v in [p5["series"][s]]), "evidence": "reports/P4,P5,P6"},
            "all_seven": p1["pass"] and p4["pass"] and p5["pass"],
            "note": "数学等价（第 3 项）不等于检测规避成功；第 5 项以持续轨迹 30 轮检测为准。",
        }

    def _final_md(self, out: dict[str, Any]) -> str:
        j = out.get("judgments", {})
        lines = ["# E11 Task2 最终报告（" + time.strftime("%Y-%m-%d %H:%M") + "）", "", "## 设计 §10 七项判定", "",
                 "| # | 判定 | 结果 |", "|---|---|---|"]
        lines += [f"| {k} | {'通过' if v.get('pass') else '未通过'} | {v.get('evidence', '')} |" for k, v in j.items() if k[:1].isdigit()]
        lines += ["", "## 聚合防御（正式轮次；行为为 dev 集华宇放行/城东误放/北港误放/正常放行）", ""]
        for s, branches in out["defenses"].items():
            for b, rounds in branches.items():
                for r, res in rounds.items():
                    tf = res["trim_fraction"]
                    lines.append(f"- {s} {b} r{r}: 裁剪比例 " + json.dumps({k: round(v, 3) for k, v in tf.items() if not k.startswith('N')}) )
                    for k in ("multi_krum_selected8", "multi_krum_selected6", "trimmed_mean"):
                        beh = res[k]["behavior"]
                        lines.append(f"    - {k}: 排除={res[k].get('excluded', '—（坐标裁剪）')} 华宇 {beh['Y_huayu_release']['hits']}/{beh['Y_huayu_release']['n']} "
                                     f"城东 {beh['FPR_chengdong']['hits']} 北港 {beh['FPR_beigang']['hits']} 正常 {beh['Acc_normal']['hits']}/{beh['Acc_normal']['n']}")
        lines += ["", "## 独立锁定确认集（最终轮）", "", "| 系列 | 分支 | 华宇放行 | 城东误放 | 北港误放 | 南站误放 | 正常 | Pair | 闭集 |", "|---|---|---|---|---|---|---|---|---|"]
        for s, branches in out["locked_confirmation"].items():
            for b, m in branches.items():
                lines.append(f"| {s} | {b} | {m['Y_huayu_release']['hits']}/{m['Y_huayu_release']['n']} | {m['FPR_chengdong']['hits']} | {m['FPR_beigang']['hits']} | {m['FPR_nanzhan']['hits']} | {m['Acc_normal']['hits']}/{m['Acc_normal']['n']} | {m['Pair_exact']['hits']} | {m['closedset']['option']['hits']}/{m['closedset']['option']['n']} |")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ 子进程入口
    def traj(self, series: str, branch: str, gpu: int, lambda_ab: tuple[float, float] | None) -> None:
        rt = self.runtime(series, gpu)
        runner = TrajectoryRunner(rt, branch, lambda_ab=lambda_ab)
        res = runner.run()
        print(json.dumps(res), flush=True)
        rt.release_model()

    def shadow(self, series: str, rounds: list[int], gpu: int) -> None:
        rt = self.runtime(series, gpu)
        sr = ShadowRunner(rt)
        for r in rounds:
            man = sr.run_round(r)
            print(f"[shadow:{series}] r{r} U trained; scales_U={man['scales_U']}", flush=True)
        rt.release_model()

    def search(self, series: str, rounds: list[int], gpu: int) -> None:
        thr = json.loads((self.art / "detection_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
        rt = self.runtime(series, gpu)
        for r in rounds:
            t0 = time.time()
            unit = ShadowSearcher(rt, thr, r).run()
            print(f"[search:{series}] r{r}: {unit['n_pass']}/{unit['n_candidates']} pass; feasible={unit['unit_feasible']} ({time.time()-t0:.0f}s)", flush=True)
        rt.release_model()

    def shadow_eval(self, series: str, rounds: list[int], lambda_ab: tuple[float, float], gpu: int) -> None:
        rt = self.runtime(series, gpu)
        sr = ShadowRunner(rt)
        history: dict[str, list] = {}
        for r in rounds:
            man = sr.construct(r, lambda_ab)
            sr.evaluate(r, lambda_ab)
            rows, history = sr.detection_rows(r, lambda_ab, scaling=self.scaling(series), history=history)
            atomic_json(Path(man["split_dir"]) / "detection_rows.json", {b: {k: v for k, v in row.items() if k != "pairwise_cosine"} for b, row in rows.items()}, overwrite=True)
            print(f"[shadow_eval:{series}] r{r}: identity pass={man['identity']['pass']}; 11-vs-U ba_rel={man['identity']['aggregate_11_vs_U']['ba_rel_err']:.2e}", flush=True)
        rt.release_model()


# ====================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["TRAJ", "SHADOW", "SEARCH", "SHADOW_EVAL", "STATUS", "PLAN"])
    ap.add_argument("--series", default=None)
    ap.add_argument("--branch", default=None, choices=list(BRANCHES))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--rounds", default=None)
    ap.add_argument("--lambda", dest="lambda_ab", default=None, help="lambda_a,lambda_b")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    proto = Protocol()
    gpus = [int(x) for x in args.gpus.split(",")]
    rounds = [int(x) for x in args.rounds.split(",")] if args.rounds else None
    lab = tuple(float(x) for x in args.lambda_ab.split(",")) if args.lambda_ab else None
    if args.stage == "P0":
        proto.p0()
    elif args.stage == "P1":
        proto.p1(gpus, gate_only=args.gate_only)
    elif args.stage == "P2":
        proto.p2()
    elif args.stage == "P3":
        proto.p3(gpus)
    elif args.stage == "P4":
        proto.p4(gpus, gate_only=args.gate_only)
    elif args.stage == "P5":
        proto.p5(gpus, gate_only=args.gate_only)
    elif args.stage == "P6":
        proto.p6(args.gpu)
    elif args.stage == "TRAJ":
        if not (args.series and args.branch):
            raise SystemExit("TRAJ needs --series and --branch")
        proto.traj(args.series, args.branch, args.gpu, lab)
    elif args.stage == "SHADOW":
        proto.shadow(args.series, rounds, args.gpu)
    elif args.stage == "SEARCH":
        proto.search(args.series, rounds, args.gpu)
    elif args.stage == "SHADOW_EVAL":
        proto.shadow_eval(args.series, rounds, lab, args.gpu)
    elif args.stage == "STATUS":
        print(json.dumps(proto.state, ensure_ascii=False, indent=1))
    elif args.stage == "PLAN":
        s = proto.series_list(); la = proto.lambda_frozen() if (proto.art / "split_selection.json").exists() else (None, None)
        plan = {"P1": [{"series": x, "branch": b} for x in s for b in ("00", "U")],
                "P3_shadow_search": [{"series": x, "rounds": proto.run["evaluation"]["search_rounds"]} for x in proto.run["series"]["development"]],
                "P4": [{"series": x, "branch": b, "lambda": la} for x in s for b in ("10", "01")],
                "P5": [{"series": x, "branch": "11", "lambda": la} for x in s],
                "trained_tasks_per_round": {b: len(alg.training_plan(b)) for b in BRANCHES},
                "series_roles": proto.run["series"]}
        print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
