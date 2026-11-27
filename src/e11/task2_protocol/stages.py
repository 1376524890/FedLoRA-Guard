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
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.e11.task2_protocol import algebra as alg
from src.e11.task2_protocol.config import ROOT, load_all, resolve_path, sha256_file
from src.e11.task2_protocol.defenses import multi_krum_defense, trimmed_mean_defense
from src.e11.task2_protocol.detection import (SCALAR_METHODS, calibration_scores, evaluate_method,  # noqa: E402
                                              fit_threshold, run_trajectory_detection)
from src.e11.task2_protocol.engine import BRANCHES, SeriesRuntime, TrajectoryRunner, git_head  # noqa: E402
from src.e11.task2_protocol.evaluation import full_eval  # noqa: E402
from src.e11.task2_protocol.inputs import Task1Inputs, audit_all, state_digest  # noqa: E402
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


def _ci_lower_positive(boot: dict) -> bool:
    """研究要求正向交互：95% CI 必须存在、有限且下界严格 > 0。
    区间整体为负（ci[1] < 0）同样"不含零"，但方向相反，不得判为通过。"""
    ci = boot.get("ci")
    if not ci or len(ci) != 2:
        return False
    lo, hi = float(ci[0]), float(ci[1])
    return math.isfinite(lo) and math.isfinite(hi) and lo > 0.0


def gate_interaction(summary_primary: dict, acc: dict) -> dict[str, Any]:
    a = acc["interaction"]
    iy, ib = summary_primary["I_Y_release"], summary_primary["I_binary_release"]
    checks = {"I_Y_gt_min": (iy["mean"] or 0.0) > a["I_Y_min"],
              "I_Y_ci_lower_gt_zero": _ci_lower_positive(iy) if a["I_Y_ci_excludes_zero"] else True,
              "I_binary_ge_min_pp": (ib["mean"] or 0.0) * 100.0 >= a["I_binary_min_pp"],
              "I_binary_ci_lower_gt_zero": _ci_lower_positive(ib) if a["I_binary_ci_excludes_zero"] else True}
    return {"checks": checks, "pass": all(checks.values()),
            "point_estimates": {"I_Y": iy.get("mean"), "I_binary_pp": None if ib.get("mean") is None else ib["mean"] * 100.0},
            "ci": {"I_Y": iy.get("ci"), "I_binary": ib.get("ci")}}


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
    def __init__(self, cfg_all: dict[str, Any] | None = None, *, allow_state_refresh: bool = False,
                 art_root: Path | None = None) -> None:
        self.cfg = cfg_all or load_all()
        run = self.cfg["run"]
        self.run = run
        # art_root 仅允许测试重定向到临时目录；正式运行一律使用配置路径
        self.art = Path(art_root) if art_root is not None else resolve_path(run["paths"]["artifacts_root"]) / run["run_id"]
        self.rep = resolve_path(run["paths"]["reports_root"]) / run["run_id"]
        self.log_root = resolve_path(run["paths"]["logs_root"])
        for p in (self.art, self.rep, self.log_root):
            p.mkdir(parents=True, exist_ok=True)
        self.state_path = self.art / "run_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "run_id": run["run_id"], "stages": {}, "protocol_content_hash": self.cfg["hashes"]["protocol_content"],
            "protocol_file_sha256": self.cfg["hashes"]["protocol_file"], "run_file_sha256": self.cfg["hashes"]["run_file"],
            "tokenizer_files": self.tokenizer_file_hashes(), "model_files": self.model_file_hashes()}
        if self.state.get("protocol_content_hash") != self.cfg["hashes"]["protocol_content"]:
            raise ConfigurationError("Protocol content changed since run started")
        # Tokenizer / 模型文件身份：训练与评价过程中变化即终止（设计 §8-P0、执行约束 require_manifest_hashes）。
        # allow_state_refresh 仅 P0 使用：P0 是审计起点，允许为旧版（无 Tokenizer/模型身份字段）状态文件补齐身份，
        # 且只允许在尚无任何轨迹阶段（P1+）通过时发生；其他阶段一律严格校验。
        tok, mdl = self.tokenizer_file_hashes(), self.model_file_hashes()
        if allow_state_refresh and not any(self.status(x) == "passed" for x in ("P1", "P2", "P3", "P4", "P5", "P6")):
            if self.state.get("tokenizer_files") != tok or self.state.get("model_files") != mdl:
                print("[Protocol] run_state.json has stale/missing tokenizer/model identity and no trajectory stage passed; refreshing identity", flush=True)
            self.state["tokenizer_files"], self.state["model_files"] = tok, mdl
            self.state["protocol_file_sha256"] = self.cfg["hashes"]["protocol_file"]
            self.state["run_file_sha256"] = self.cfg["hashes"]["run_file"]
            self.save()
        else:
            if self.state.get("tokenizer_files") != tok:
                raise ConfigurationError("Tokenizer files changed since run started")
            if self.state.get("model_files") != mdl:
                raise ConfigurationError("Model files changed since run started")
        self.state["last_stage_invocation"] = {"ts": time.time(), "git": git_head()["commit"]}

    def tokenizer_file_hashes(self) -> dict[str, str]:
        """全部系列的模型目录相同（配置 paths.model_dir 唯一）；取首个系列的 Tokenizer 文件哈希。"""
        return Task1Inputs(self.cfg, self.series_list()[0]).tokenizer_file_hashes()

    def model_file_hashes(self) -> dict[str, str]:
        inp = Task1Inputs(self.cfg, self.series_list()[0])
        return {p.name: sha256_file(p) for p in sorted(inp.model_dir.glob("*.safetensors"))} | {"config.json": sha256_file(inp.model_dir / "config.json")}

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
    TERM_WAIT_S = 30   # SIGTERM 后等待秒数；超时仍未退出则 SIGKILL

    def _spawn(self, g: int, job: dict[str, Any]) -> tuple[dict[str, Any], subprocess.Popen, Any, str]:
        tag = f"{job['stage']}_{job['series']}_{job.get('branch', job.get('rounds', ''))}".replace(",", "-").replace(" ", "")
        log_path = self.log_root / f"{tag}_gpu{g}.log"
        log = open(log_path, "a")
        cmd = [PY, "-m", "src.e11.task2_protocol.stages", "--stage", job["stage"], "--series", job["series"], "--gpu", str(g)]
        if job.get("branch"):
            cmd += ["--branch", job["branch"]]
        if job.get("rounds"):
            cmd += ["--rounds", ",".join(str(x) for x in job["rounds"])]
        if job.get("lambda"):
            cmd += ["--lambda", f"{job['lambda'][0]},{job['lambda'][1]}"]
        print(f"[queue] {tag} -> gpu{g} ({log_path})", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return job, proc, log, str(log_path)

    @staticmethod
    def _terminate_all(running: dict[int, tuple[dict[str, Any], subprocess.Popen, Any, str]], wait_s: int) -> list[dict[str, Any]]:
        """统一终止全部子进程：先 SIGTERM，等待 wait_s 秒，仍未退出者 SIGKILL；最后等待全部退出。
        返回每个被终止进程的 {gpu, job, exit_code, log}。不吞掉任何异常以外的信息。"""
        outcomes = []
        for g, (job, proc, log, log_path) in running.items():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)   # 子进程为独立会话（start_new_session），整组终止
                except ProcessLookupError:
                    pass                              # 已在 poll 与 kill 之间退出
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if all(proc.poll() is not None for _, proc, _, _ in running.values()):
                break
            time.sleep(0.5)
        for g, (job, proc, log, log_path) in running.items():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for g, (job, proc, log, log_path) in running.items():
            code = proc.wait()
            log.close()
            outcomes.append({"gpu": g, "job": job, "exit_code": code, "log": log_path})
        return outcomes

    def run_queue(self, jobs: list[dict[str, Any]], gpus: list[int]) -> None:
        """jobs: {stage, series, [branch], [rounds], [lambda]}；按 GPU 轮转；统一进程管理：
        记录全部子进程、持续轮询全部退出状态；任一非零退出 → 停止调度、SIGTERM→SIGKILL 其余进程、
        等待全部退出、记录失败 GPU/任务/退出码/日志 → 以非零码退出，不进入后续阶段。
        用户中断（KeyboardInterrupt/SIGTERM）走同一清理路径，原异常原样上抛。"""
        queues: dict[int, list] = {g: [] for g in gpus}
        for i, j in enumerate(jobs):
            queues[gpus[i % len(gpus)]].append(j)
        running: dict[int, tuple[dict[str, Any], subprocess.Popen, Any, str]] = {}
        failure: dict[str, Any] | None = None
        try:
            for g in gpus:
                if queues[g]:
                    running[g] = self._spawn(g, queues[g].pop(0))
            while running and failure is None:
                for g in list(running):
                    job, proc, log, log_path = running[g]
                    code = proc.poll()
                    if code is None:
                        continue
                    log.close()
                    del running[g]
                    if code != 0:
                        failure = {"gpu": g, "job": job, "exit_code": code, "log": log_path}
                        break
                    print(f"[queue] {job['series']} {job['stage']} {job.get('branch', job.get('rounds'))} 完成（gpu{g}）", flush=True)
                    if queues[g]:
                        running[g] = self._spawn(g, queues[g].pop(0))
                if failure is None:
                    time.sleep(15)
        except BaseException:
            # 用户中断或调度异常：同样终止全部子进程，再原样上抛
            terminated = self._terminate_all(running, self.TERM_WAIT_S)
            for t in terminated:
                print(f"[queue] 中断后终止 gpu{t['gpu']} {t['job']} exit={t['exit_code']} log={t['log']}", flush=True)
            raise
        if failure is not None:
            terminated = self._terminate_all(running, self.TERM_WAIT_S)
            for t in terminated:
                print(f"[queue] 因失败而终止 gpu{t['gpu']} {t['job']} exit={t['exit_code']} log={t['log']}", flush=True)
            detail = {**failure, "terminated_siblings": terminated}
            self.mark(failure["job"]["stage"], "failed", error=f"{failure['job']} exited {failure['exit_code']}", detail=detail)
            raise SystemExit(f"[stop] 子任务失败 exit={failure['exit_code']}: gpu{failure['gpu']} {failure['job']}"
                             f"（日志 {failure['log']}）；已终止 {len(terminated)} 个其余子任务")

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
        lines += ["## 包源码哈希（src/e11/task2_protocol/*.py）", ""] + [f"- `{k}`：`{v}`" for k, v in manifest["package_sources"].items()] + [""]
        for series, info in manifest["series"].items():
            lines += [f"## {series}", "", f"- T0：`{info['t0']['path']}` sha256 `{info['t0']['sha256']}`，state_digest `{info['t0']['state_digest'][:16]}`",
                      f"- 模型文件：" + "；".join(f"`{k}`=`{v[:16]}…`" for k, v in info["model_files"].items()),
                      f"- Tokenizer：类型 `{info['tokenizer']['class']}`；文件 " + "；".join(f"`{k}`=`{v['sha256'][:16]}…`（{v['bytes']} B）" for k, v in info["tokenizer"]["files"].items())
                      + f"；特殊词元 pad={info['tokenizer']['pad_token_id']} eos={info['tokenizer']['eos_token_id']}；chat template sha256 `{(info['tokenizer']['chat_template_sha256'] or 'none')[:16]}`",
                      "", "| 检查 | 结果 | 备注 |", "|---|---|---|"]
            lines += [f"| {c['name']} | {'✅' if c['ok'] else '❌'} | {c['detail'][:160]} |" for c in info["checks"]]
            lines.append("")
        lines += ["## 历史结果复用判定（设计 §7）", "", "| 历史内容 | 复用 | 说明 |", "|---|---|---|"]
        lines += [f"| {k} | {'是' if v['reuse'] else '否'} | {v.get('as', v.get('reason', ''))} |" for k, v in manifest["reuse_decisions"].items()]
        (rep / "INPUT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        passed = bool(manifest["passed"] and not manifest["git"]["dirty_tracked"])
        self.mark("P0", "passed" if passed else "failed", report=str(rep / "INPUT_AUDIT.md"),
                  git=manifest["git"], package_sources=manifest["package_sources"])
        if not passed:
            why = [] if manifest["passed"] else ["input audit checks failed"]
            if manifest["git"]["dirty_tracked"]:
                why.append("tracked files have uncommitted modifications (P0 must run on a committed tree)")
            raise SystemExit(f"[stop] P0 输入审计未通过：{why}（见 reports P0/INPUT_AUDIT.md）")
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
                s00 = load_eval(self._trajectory_dir(s, "00"), r)
                su = load_eval(self._trajectory_dir(s, "U"), r)
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
                run_trajectory_detection(self._trajectory_dir(s, b), s, b, scaling=self.scaling(s),
                                         load_flat=rt.load_flat, sha256_file=sha256_file)
            rows_by_series[s] = [json.loads(l) for l in (self._trajectory_dir(s, "00") / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        # 校准：开发种子 00 轨迹正常上传；有效轮次规则与 evaluate_method 完全一致（detection.valid_rows）
        fail_on_degenerate = bool(self.cfg["protocol"]["detection"]["fail_on_foolsgold_degenerate_equal_weights"])
        cal_scores = {m: [] for m in SCALAR_METHODS}
        cal_invalid = {m: [] for m in SCALAR_METHODS}
        for s in self.run["series"]["development"]:
            for m in SCALAR_METHODS:
                vals, bad = calibration_scores(rows_by_series[s], m)
                cal_scores[m] += vals
                cal_invalid[m] += [{"series": s, **b} for b in bad]
        fpr = float(self.cfg["protocol"]["detection"]["false_positive_working_point"])
        thr = {m: fit_threshold(cal_scores[m], fpr) for m in SCALAR_METHODS}
        for m in SCALAR_METHODS:  # 协议 fail_on_foolsgold_degenerate_equal_weights：校准集不得含 degenerate 轮
            if cal_invalid[m] and not fail_on_degenerate:
                raise ConfigurationError("protocol says fail_on_foolsgold_degenerate_equal_weights=false, "
                                         "but invalid calibration rounds exist; this is not supported")
        thresholds = {"thresholds": {m: t["threshold"] for m, t in thr.items()}, "fit": thr, "target_fpr": fpr,
                      "calibration_source": "traj_00 normal uploads of development series, valid rounds only (detection.valid_rows)",
                      "calibration_series": list(self.run["series"]["development"]),
                      "calibration_invalid_rounds": cal_invalid,
                      "fail_on_foolsgold_degenerate_equal_weights": fail_on_degenerate,
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
                rows_u = [json.loads(l) for l in (self._trajectory_dir(s, "U") / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                mal = {row["round"]: ["U"] for row in rows_u}
                rep_s["U_detection"][m] = evaluate_method(rows_u, mal, thresholds["thresholds"][m], m)
            rep_s["cbdf_00"] = json.loads((self._trajectory_dir(s, "00") / "detection_cbdf.json").read_text(encoding="utf-8"))["detected_clients"]
            rep_s["cbdf_U"] = json.loads((self._trajectory_dir(s, "U") / "detection_cbdf.json").read_text(encoding="utf-8"))["detected_clients"]
            report["series"][s] = rep_s
        atomic_json(rep / "DETECTION.json", report, overwrite=True)
        lines = ["# P2 检测报告（阈值由开发种子 00 轨迹正常上传校准，冻结）", "",
                 "阈值：" + json.dumps(thresholds["thresholds"]), "",
                 "校准有效轮次规则：" + ("degenerate 轮**排除**（协议 fail_on_foolsgold_degenerate_equal_weights=true）" if fail_on_degenerate else "未排除"),
                 "校准排除轮次：" + json.dumps(cal_invalid, ensure_ascii=False), "",
                 "| 系列 | 角色 | 方法 | 00 有效轮 | 00 正常 FPR | U 有效轮/总轮 | U 检出率(recall) | U Precision | U ROC-AUC | U PR-AUC | U 平均名次 | U 无效轮 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, rs in report["series"].items():
            for m in SCALAR_METHODS:
                f, d = rs["normal_clients_FPR_00"][m], rs["U_detection"][m]
                ranks = [v["ranks"]["U"] for v in d["per_round"].values()]
                lines.append(f"| {s} | {rs['role']} | {m} | {f['rounds_scored']} | {f['fpr']:.3f} | {d['rounds_scored']}/{d['rounds_total']} | {d['recall']:.3f} | {d['precision']:.3f} | {d['roc_auc']} | {d['pr_auc']} | {sum(ranks)/max(len(ranks),1):.2f} | {[x['round'] for x in d['invalid_rounds']]} |")
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

    def _trajectory_dir(self, series: str, branch: str) -> Path:
        """轨迹目录的唯一构造入口：traj_<branch>（00 / U / 10 / 01 / 11）。"""
        return self.art / series / f"traj_{branch}"

    def _detection_evasion(self, s: str, branch: str, malicious: list[str]) -> dict[str, Any]:
        """一条轨迹的检测规避结果。每 (series, branch) 只执行一次 run_trajectory_detection（重算全部轮次 + CBDF），
        三种方法（Norm/Cosine/FoolsGold）全部评分；判定 = 有效轮数充足 且 A/B 被标记率 ≤ evasion_max_flag_rate。
        FoolsGold degenerate 轮按 detection.valid_rows 统一规则排除并记录原因——被排除的轮次既不是"被标记"
        也不是"未被标记"；有效轮数不足时不判定规避成立（pass=False）。"""
        tdir = self._trajectory_dir(s, branch)
        run_trajectory_detection(tdir, s, branch, scaling=self.scaling(s),
                                 load_flat=self.runtime(s, 0).load_flat, sha256_file=sha256_file)
        rows = [json.loads(l) for l in (tdir / "detection_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        thr = json.loads((self.art / "detection_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
        ev_acc = self.run["acceptance"]["detection_evasion"]
        max_rate = float(ev_acc["evasion_max_flag_rate"])
        required = list(ev_acc["require_ab_unflagged_by_all"])
        if sorted(required) != sorted(SCALAR_METHODS):
            raise ConfigurationError(f"acceptance.detection_evasion.require_ab_unflagged_by_all must be exactly {SCALAR_METHODS}")
        out = {}
        for m in SCALAR_METHODS:
            ev = evaluate_method(rows, {row["round"]: malicious for row in rows}, thr[m], m)
            scored = ev["rounds_scored"]
            sufficient = bool(scored) and ev["sufficient_valid_rounds"]
            rate = ev["recall"] if scored else None   # 没有有效轮次时不得记 0.0（会被误读为"未被标记"）
            out[m] = {"flag_rate": rate, "rounds_scored": scored, "rounds_total": ev["rounds_total"],
                      "fpr_normal": ev["fpr"], "threshold": thr[m], "malicious": list(malicious),
                      "invalid_rounds": ev["invalid_rounds"], "sufficient_valid_rounds": sufficient,
                      "per_round_flagged": {k: {n: v["scores"][n] > thr[m] for n in malicious if n in v["scores"]}
                                            for k, v in ev["per_round"].items()},
                      "pass": bool(sufficient and rate is not None and rate <= max_rate)}
        return out

    def _scale_band_check(self, s: str, branch: str, role: str, rounds: list[int]) -> dict[str, Any]:
        """尺度门（正式轮次）：A/B 三种尺度 / 同轮正常客户端中位数 ∈ scale_band。路径经 _trajectory_dir 构造。"""
        band = [float(x) for x in self.run["acceptance"]["scale_band"]]
        rt = self.runtime(s, 0)
        scale_ok, scale_detail = True, {}
        for r in rounds:
            man = json.loads((self._trajectory_dir(s, branch) / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
            G = alg.to_pairs(rt.load_flat(Path(man["start"]["path"])))
            normals = [alg.three_scales(alg.to_pairs(rt.load_flat(Path(man["uploads"][n]["path"]))), G, self.scaling(s))
                       for n in man["slots"] if n.startswith("N")]
            med = {k: float(sorted(x[k] for x in normals)[len(normals) // 2]) for k in ("N_eff_BA", "F_P", "F_Q")}
            ratios = {k: man["constructs"][role][k] / med[k] for k in med}
            ok = all(band[0] <= v <= band[1] for v in ratios.values())
            scale_detail[str(r)] = {"ratios": ratios, "median_normal": med, "band": band, "pass": ok}
            scale_ok &= ok
        return {"pass": scale_ok, "per_round": scale_detail}

    def _gate_p4(self) -> dict[str, Any]:
        """P4 单支验收。每个系列六项检查：A 行为弱、B 行为弱、A 尺度在带内、B 尺度在带内、A 检测规避、B 检测规避。
        行为/尺度按正式指定轮次逐一判定；检测对每条轨迹只执行一次（A 在 10 分支、B 在 01 分支）。
        任一系列任一项失败 → P4 失败。"""
        acc = self.run["acceptance"]
        rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        out: dict[str, Any] = {"rounds": rounds, "series": {}}
        for s in self.series_list():
            sd = self.art / s
            per = {}
            for b, role in (("10", "A"), ("01", "B")):
                tdir = self._trajectory_dir(s, b)
                behavior = {str(r): gate_single_weak(load_eval(tdir, r), acc) for r in rounds}
                per[f"{role}_behavior_weak_in_{b}"] = {"pass": all(v["pass"] for v in behavior.values()), "per_round": behavior}
                per[f"{role}_scale_band_in_{b}"] = self._scale_band_check(s, b, role, rounds)
                methods = self._detection_evasion(s, b, [role])          # 单次检测
                per[f"{role}_detection_evasion_in_{b}"] = {"pass": all(v["pass"] for v in methods.values()), "methods": methods}
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

    def _shadow_math_checks(self, s: str, r: int, lambda_frozen: tuple[float, float]) -> dict[str, Any]:
        """影子轮次 r 的数学验收（设计 §4 阶段五、§5）。独立重新读取磁盘产物核验，不只信任 split_manifest.identity.pass。
        七项：(1) A+B 因子加和恒等式 (2) 真实 FedAvg 后 11 与 U 的因子一致性 (3) 有效 BA 权重一致性
        (4) 四分支线性可加性 (5) C_W 显式差分 vs 低秩交叉项（两种误差口径分别记录，门槛只对应 e_W）
        (6) 文件 SHA256 与 manifest 一致 (7) 实际使用的 λ 与 P3 冻结 λ 完全一致。
        manifest 缺失 → 全部检查记为失败（不抛异常，由 P5 汇总判定）；数值超门槛 → 对应项失败。"""
        sdir = self.art / s / "shadow" / f"round_{r:03d}" / split_tag(lambda_frozen)
        smp = sdir / "split_manifest.json"
        num = self.run["acceptance"]["numerical"]
        tol = {"ab_sum_rel_max": 1e-12, "factor_max_abs": float(num["factor_max_abs"]),
               "ba_rel_err_max": float(num["ba_rel_err_max"]), "linear_additivity_abs_max": float(num["factor_max_abs"]),
               "interaction_e_W_max": float(num["interaction_abs_diff_over_W_max"])}
        checks: dict[str, Any] = {}
        if not smp.exists():
            return {"pass": False, "split_dir": str(sdir), "missing_manifest": str(smp), "tolerance": tol,
                    "checks": {k: {"pass": False, "error": f"split_manifest.json missing: {smp}"} for k in
                               ("ab_sum_identity", "aggregate_11_vs_U_factors", "effective_ba", "linear_additivity",
                                "C_W_explicit_vs_lowrank", "file_sha256", "frozen_lambda")}}
        man = json.loads(smp.read_text(encoding="utf-8"))
        rt = self.runtime(s, 0)
        sc = self.scaling(s)
        # (7) λ 冻结一致性：manifest 记录的 λ、目录名中的 λ 必须都等于 P3 冻结值（精确相等）
        lam_ok = (man.get("lambda_ab") == [lambda_frozen[0], lambda_frozen[1]]
                  and man["identity"].get("lambda_a") == lambda_frozen[0]
                  and man["identity"].get("lambda_b") == lambda_frozen[1]
                  and sdir.name == split_tag(lambda_frozen))
        checks["frozen_lambda"] = {"pass": bool(lam_ok), "manifest_lambda_ab": man.get("lambda_ab"),
                                   "frozen_lambda_ab": list(lambda_frozen), "split_dir_name": sdir.name}
        # (6) 文件 SHA256：manifest 列出的每个文件（A/B/四分支聚合/影子 U/冻结起点/构造输入）逐个重算
        expected = {"A": man["A"], "B": man["B"], "frozen_reference": man["frozen_reference"]}
        expected.update({f"aggregate_{b}": man["aggregates"][b] for b in ("00", "10", "01", "11", "U")})
        expected.update({f"input_{k}": v for k, v in man["construction_inputs"].items()})
        sha_detail, sha_ok = {}, True
        for key, ent in expected.items():
            path = Path(ent["path"])
            actual = sha256_file(path) if path.exists() else None
            ok = actual is not None and actual == ent["sha256"]
            sha_ok &= ok
            sha_detail[key] = {"path": str(path), "expected": ent["sha256"], "actual": actual, "pass": ok}
        checks["file_sha256"] = {"pass": bool(sha_ok), "files": sha_detail}
        if not sha_ok:
            # 哈希不一致：不得继续用可能被篡改的张量做数值核验
            for k in ("ab_sum_identity", "aggregate_11_vs_U_factors", "effective_ba", "linear_additivity", "C_W_explicit_vs_lowrank"):
                checks[k] = {"pass": False, "error": "skipped numeric verification because file sha256 mismatch"}
            return {"pass": False, "split_dir": str(sdir), "tolerance": tol, "checks": checks}
        # 从磁盘回读张量做独立数值核验
        received = rt.load_flat(Path(man["frozen_reference"]["path"]))
        G = alg.to_pairs(received)
        U = alg.to_pairs(rt.load_flat(Path(man["construction_inputs"]["U"]["path"])))
        N1 = alg.to_pairs(rt.load_flat(Path(man["construction_inputs"]["N1"]["path"])))
        A = alg.to_pairs(rt.load_flat(Path(man["A"]["path"])))
        B = alg.to_pairs(rt.load_flat(Path(man["B"]["path"])))
        p_w = float(self.cfg["protocol"]["protocol"]["client_weight"])
        # (1) A+B 因子加和恒等式（对回读的 FP32 A/B 升精度计算；FP32 舍入余差以 factor_max_abs 为准）
        ab64 = alg.ab_sum_identity_error(G, U, N1, alg.to_fp64(A), alg.to_fp64(B), p_w)
        ab_rec = man["identity"]["ab_sum_identity_fp64"]
        checks["ab_sum_identity"] = {"pass": max(ab_rec.values()) <= tol["ab_sum_rel_max"],
                                     "fp64_construct_rel_err": ab_rec, "serialized_fp32_pair_sum_rel_err": ab64}
        # (2)(3) 真实 FedAvg 后 11 与 U：因子最大绝对误差、有效 BA 相对误差（回读聚合文件）
        P = {b: alg.to_pairs(rt.load_flat(Path(man["aggregates"][b]["path"]))) for b in ("00", "10", "01", "11", "U")}
        cmp = alg.compare_aggregates(f"{s} r{r} 11_vs_U", P["11"], P["U"], sc,
                                     ba_tol=tol["ba_rel_err_max"], factor_tol=tol["factor_max_abs"])
        checks["aggregate_11_vs_U_factors"] = {"pass": bool(cmp["factor_pass"]), "factor_max_abs": cmp["factor_max_abs"],
                                               "tol": tol["factor_max_abs"], "recorded": man["identity"]["aggregate_11_vs_U"]["factor_max_abs"]}
        checks["effective_ba"] = {"pass": bool(cmp["ba_pass"]), "ba_rel_err": cmp["ba_rel_err"], "tol": tol["ba_rel_err_max"],
                                  "ba_module_max_abs": cmp["ba_module_max_abs"], "ba_module_ref_norm": cmp["ba_module_ref_norm"],
                                  "recorded": man["identity"]["aggregate_11_vs_U"]["ba_rel_err"]}
        # (4) 四分支线性可加性（因子层面）
        lin = alg.linear_additivity_check(P["00"], P["10"], P["01"], P["11"])
        checks["linear_additivity"] = {"pass": max(lin.values()) <= tol["linear_additivity_abs_max"], **lin,
                                       "tol": tol["linear_additivity_abs_max"],
                                       "recorded": man["identity"]["linear_additivity_P11_minus_(P10+P01-P00)"]}
        # (5) C_W 两路径：独立重算（e_W 门槛；e_C 仅记录）
        cw = alg.compare_interaction_paths(P["00"], P["10"], P["01"], P["11"], sc)
        e_w = cw["abs_diff_over_max_W"]
        checks["C_W_explicit_vs_lowrank"] = {
            "pass": bool(e_w is not None and e_w <= tol["interaction_e_W_max"]),
            "e_W_abs_diff_over_max_W": e_w, "e_W_tol": tol["interaction_e_W_max"],
            "e_C_rel_diff_over_C_lowrank_RECORD_ONLY": cw["rel_diff_over_C_lowrank"], "e_C_note": cw["rel_diff_over_C_note"],
            "abs_diff": cw["abs_diff"], "max_W_norm": cw["max_W_norm"], "total_explicit_norm": cw["total_explicit_norm"],
            "total_lowrank_norm": cw["total_lowrank_norm"], "max_trace_rel_diff": cw["max_trace_rel_diff"],
            "error_definitions": cw["error_definitions"],
            "recorded_e_W": man["identity"]["C_W_paths"].get("abs_diff_over_max_W")}
        return {"pass": all(v["pass"] for v in checks.values()), "split_dir": str(sdir), "tolerance": tol, "checks": checks}

    def _persistent_11_vs_U(self, s: str) -> dict[str, Any]:
        """持续轨迹逐轮：真实 FedAvg 后 traj_11.global_r 与 traj_U.global_r 的因子/BA 一致性（回读文件）。"""
        rt = self.runtime(s, 0)
        num = self.run["acceptance"]["numerical"]
        per_round, first_fail = {}, None
        for r in range(1, int(self.cfg["protocol"]["protocol"]["rounds"]) + 1):
            a11 = alg.to_pairs(rt.load_flat(self._trajectory_dir(s, "11") / f"global_{r:03d}.safetensors"))
            au = alg.to_pairs(rt.load_flat(self._trajectory_dir(s, "U") / f"global_{r:03d}.safetensors"))
            c = alg.compare_aggregates(f"r{r}", a11, au, self.scaling(s), ba_tol=num["ba_rel_err_max"], factor_tol=num["factor_max_abs"])
            per_round[str(r)] = {"pass": c["pass"], "factor_max_abs": c["factor_max_abs"], "ba_rel_err": c["ba_rel_err"]}
            if not c["pass"] and first_fail is None:
                first_fail = r
        worst_f = max(max(v["factor_max_abs"]["P"], v["factor_max_abs"]["Q"]) for v in per_round.values())
        worst_b = max(v["ba_rel_err"] for v in per_round.values())
        return {"pass": first_fail is None, "first_failed_round": first_fail, "worst_factor_max_abs": worst_f,
                "worst_ba_rel_err": worst_b, "tolerance": {"factor_max_abs": num["factor_max_abs"], "ba_rel_err_max": num["ba_rel_err_max"]},
                "per_round": per_round}

    def _gate_p5(self) -> dict[str, Any]:
        """P5 联合验收。每个系列：
        ① 持续 11 联合行为达 U 强行为门（正式轮次）；② 持续 11 ≡ 持续 U 全部 30 轮；
        ③ 持续 11 中 A、B 的检测规避（Norm/Cosine/FoolsGold）；④ 每个影子轮次七项数学核验；
        ⑤ 每个影子轮次四分支交互门（点估计 + 95% CI 下界 > 0）；轨迹层面差分仅报告（非同轮因果）。"""
        acc = self.run["acceptance"]
        rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        inter_rounds = [r for r in rounds if r in [int(x) for x in self.run["evaluation"]["shadow_rounds"]]]
        n_boot = int(self.cfg["protocol"]["statistics"]["bootstrap_replicates"])
        level = float(self.cfg["protocol"]["statistics"]["confidence_level"])
        seed = n_boot * 100003
        la, lb = self.lambda_frozen()
        out: dict[str, Any] = {"rounds": rounds, "interaction_rounds": inter_rounds, "shadow_rounds": [int(x) for x in self.run["evaluation"]["shadow_rounds"]],
                               "lambda_frozen": [la, lb], "series": {}}
        for s in self.series_list():
            chk: dict[str, Any] = {}
            # ① 联合强行为（持续 11，正式轮次）
            joint = {str(r): gate_strong(load_eval(self._trajectory_dir(s, "11"), r), load_eval(self._trajectory_dir(s, "00"), r), acc["joint_strong"]) for r in rounds}
            chk["joint_strong_persistent"] = {"pass": all(v["pass"] for v in joint.values()), "per_round": joint}
            # ② 持续 11 ≡ 持续 U（逐轮真实聚合）
            chk["aggregate_11_equals_U_all_rounds"] = self._persistent_11_vs_U(s)
            # ③ 检测规避（持续 11 中的 A 与 B；单次检测）
            ev = self._detection_evasion(s, "11", ["A", "B"])
            chk["detection_evasion_A_B_in_11"] = {"pass": all(v["pass"] for v in ev.values()), "methods": ev}
            # ④ 影子轮次七项数学核验（全部影子轮次，不只正式轮次）
            math_all = {str(r): self._shadow_math_checks(s, r, (la, lb)) for r in out["shadow_rounds"]}
            chk["shadow_math_identities"] = {"pass": all(v["pass"] for v in math_all.values()), "per_round": math_all}
            # ⑤ 同轮影子四分支交互门（同起点因果估计）
            inter = {}
            for r in inter_rounds:
                sdir = self.art / s / "shadow" / f"round_{r:03d}" / split_tag((la, lb))
                rows = {b: [json.loads(l) for l in (sdir / f"eval_dev_{b}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                        for b in SHADOW_BRANCHES}
                rows["00"] = load_eval_rows(self._trajectory_dir(s, "00"), r)
                summ = summarize_interaction(interaction_table(rows), n_boot=n_boot, seed=seed, level=level)
                atomic_json(sdir / "interaction.json", summ, overwrite=True)
                inter[str(r)] = {"gate": gate_interaction(summ["huayu_defective_primary"], acc), "summary": summ}
            chk["shadow_interaction"] = {"pass": all(v["gate"]["pass"] for v in inter.values()), "per_round": inter}
            # 轨迹层面差分（明确标注：轨迹层面，非同轮因果；仅报告，不参与判定）
            traj = {}
            for r in rounds:
                rows = {b: load_eval_rows(self._trajectory_dir(s, b), r) for b in ("00", "10", "01", "11")}
                traj[str(r)] = summarize_interaction(interaction_table(rows), n_boot=n_boot, seed=seed, level=level)
            chk["trajectory_level_four_branch_difference_NOT_same_round_causal"] = traj
            gated = {k: v["pass"] for k, v in chk.items() if isinstance(v, dict) and "pass" in v and "NOT" not in k}
            out["series"][s] = {"checks": chk, "required_checks": gated, "pass": all(gated.values())}
        out["pass"] = all(v["pass"] for v in out["series"].values())
        return out

    # ------------------------------------------------------------ P6
    LOCKED_SET_CONSUMERS = ("src/e11/task2_protocol/stages.py::Protocol.p6/_locked_confirmation_check",)

    def _locked_confirmation_check(self, s: str) -> dict[str, Any]:
        """独立锁定确认集核验（设计 §8-P6）。确认集必须：manifest 哈希一致、1536 行、与训练/开发/测试商品零隔离、
        字段完整；并且只被 P6 使用——不参与 λ 搜索、检测阈值拟合、超参数选择（代码层核对：search.py / p2 / p3 不读取该文件）。
        任何一项失败 → 返回 pass=False（P6 必须失败），不修补。"""
        rt = self.runtime(s, 0)
        lc = rt.inputs.locked_confirmation()
        out: dict[str, Any] = {"pass": bool(lc["ok"]), "path": lc.get("path"), "sha256": lc.get("sha256"), "n_rows": lc.get("n_rows"),
                               "n_products": lc.get("n_products"), "overlap_with_task1_products": lc.get("overlap_with_task1_products"),
                               "detail": lc.get("detail", "")}
        if not lc["ok"]:
            return out
        # manifest 记录的 sha256 必须与磁盘文件一致（locked_confirmation() 已算 actual，这里再独立核对 manifest 字段）
        mp = resolve_path(self.run["paths"]["locked_confirmation_manifest"])
        man = json.loads(mp.read_text(encoding="utf-8"))
        rows_path = mp.parent / "locked_confirmation_1536.jsonl"
        actual = sha256_file(rows_path)
        out["manifest_sha256"] = man["sha256"]["locked_confirmation_1536.jsonl"]
        out["sha256_matches_manifest"] = actual == man["sha256"]["locked_confirmation_1536.jsonl"]
        out["pass"] = bool(out["pass"] and out["sha256_matches_manifest"])
        # 隔离：确认集商品 vs 本系列训练对(30)、dev(1152)、test(1536) 商品
        used = {r["product_id"] for split in ("dev", "test") for r in rt.inputs.eval_rows(split)}
        used |= {q["product_id"] for q in rt.inputs.pairs.values()}
        pids = {r["product_id"] for r in lc["rows"]}
        out["isolation"] = {"n_locked_products": len(pids), "n_task1_products": len(used), "overlap": len(pids & used),
                            "train_pairs": 30, "dev_rows": len(rt.inputs.eval_rows("dev")), "test_rows": len(rt.inputs.eval_rows("test"))}
        out["pass"] = bool(out["pass"] and not (pids & used))
        # 仅 P6 使用：λ 选择与阈值拟合文件不得引用确认集路径；源码层 search.py / p2 / p3 不读取
        sel = self.art / "split_selection.json"
        if sel.exists():
            sel_txt = sel.read_text(encoding="utf-8")
            out["used_in_lambda_search"] = "locked_confirmation" in sel_txt
            out["pass"] = bool(out["pass"] and not out["used_in_lambda_search"])
        thr_p = self.art / "detection_thresholds.json"
        if thr_p.exists():
            thr_txt = thr_p.read_text(encoding="utf-8")
            out["used_in_threshold_fit"] = "locked_confirmation" in thr_txt
            out["pass"] = bool(out["pass"] and not out["used_in_threshold_fit"])
        return out

    def p6(self, gpu: int) -> None:
        self.require("P5")
        la, lb = self.lambda_frozen()
        rep = self.report_dir("P6")
        acc = self.run["acceptance"]
        gate_rounds = [int(x) for x in self.run["evaluation"]["formal_gate_rounds"]]
        final_r = int(self.cfg["protocol"]["protocol"]["rounds"])
        out: dict[str, Any] = {"defenses": {}, "locked_confirmation": {}, "locked_confirmation_check": {},
                               "judgments": {}, "gate_rounds": gate_rounds, "lambda_frozen": [la, lb]}
        for s in self.series_list():
            rt = self.runtime(s, gpu)
            rt.ensure_model()
            mk = self.cfg["protocol"]["aggregation_defenses"]["multi_krum"]
            trim = int(self.cfg["protocol"]["aggregation_defenses"]["trimmed_mean"]["trim_each_coordinate_each_side"])
            if int(mk["neighbor_count"]) != int(mk["n"]) - int(mk["f"]) - 2:
                raise ConfigurationError("multi_krum.neighbor_count must equal n - f - 2 (repo implementation)")
            out["defenses"][s] = {}
            for b in ("U", "11"):
                res_b = {}
                for r in gate_rounds:
                    man = json.loads((self._trajectory_dir(s, b) / f"round_{r:03d}" / "round_manifest.json").read_text(encoding="utf-8"))
                    names = man["slots"]
                    for n in names:   # 防御输入 = 磁盘回读且哈希核对过的真实上传
                        if sha256_file(Path(man["uploads"][n]["path"])) != man["uploads"][n]["sha256"]:
                            raise CheckpointError(f"{s} {b} r{r} upload {n} changed on disk")
                    flats = [rt.load_flat(Path(man["uploads"][n]["path"])) for n in names]
                    received = rt.load_flat(Path(man["start"]["path"]))
                    tm = trimmed_mean_defense(flats, names, trim=trim)
                    res = {"trim_fraction": tm["coordinate_trim_fraction"],
                           "trim_fraction_per_tensor_ABU": {k: {n: v for n, v in d.items() if not n.startswith("N")} for k, d in tm["coordinate_trim_fraction_per_tensor"].items()},
                           "trim_fraction_A_B_U": {n: tm["coordinate_trim_fraction"][n] for n in names if not n.startswith("N")}}
                    for sel in mk["selected_counts"]:
                        d = multi_krum_defense(flats, received, names, f=int(mk["f"]), selected=int(sel))
                        res[f"multi_krum_selected{sel}"] = {"excluded": d["excluded"], "selected": d["selected"], "neighbor_count": d["neighbor_count"],
                                                            "scores": d["scores"],
                                                            "aggregate_sha256": state_digest(d["aggregate"]),
                                                            "behavior": full_eval(rt.model, rt.tokenizer, d["aggregate"], rt.inputs.eval_rows("dev"), rt.device, rt.tcfg, "dev",
                                                                                  self._trajectory_dir(s, b) / f"round_{r:03d}" / f"defense_mkrum{sel}_dev")}
                    res["trimmed_mean"] = {"behavior": full_eval(rt.model, rt.tokenizer, tm["aggregate"], rt.inputs.eval_rows("dev"), rt.device, rt.tcfg, "dev",
                                                                 self._trajectory_dir(s, b) / f"round_{r:03d}" / "defense_trimmed_dev"),
                                           "aggregate_sha256": state_digest(tm["aggregate"])}
                    res_b[str(r)] = res
                out["defenses"][s][b] = res_b
            # 独立锁定确认集（最终轮，五分支）——先核验，不通过则 P6 失败且不写确认集评价
            out["locked_confirmation_check"][s] = self._locked_confirmation_check(s)
            if out["locked_confirmation_check"][s]["pass"]:
                rows = rt.inputs.locked_confirmation()["rows"]
                lc = {}
                for b in BRANCHES:
                    flat = rt.load_flat(self._trajectory_dir(s, b) / f"global_{final_r:03d}.safetensors")
                    lc[b] = full_eval(rt.model, rt.tokenizer, flat, rows, rt.device, rt.tcfg, "test",
                                      self._trajectory_dir(s, b) / "eval_locked_confirmation")
                    lc[b]["global_sha256"] = sha256_file(self._trajectory_dir(s, b) / f"global_{final_r:03d}.safetensors")
                out["locked_confirmation"][s] = lc
            rt.release_model()
        judgments = self._finalize_judgments(self._judgments(out))
        out["judgments"] = judgments
        passed = bool(judgments["all_seven"] and all(v["pass"] for v in out["locked_confirmation_check"].values()))
        atomic_json(rep / "DEFENSES_AND_CONFIRMATION.json", out, overwrite=True)
        atomic_json(rep / "FINAL.json", out, overwrite=True)
        (rep / "FINAL_REPORT.md").write_text(self._final_md(out, passed), encoding="utf-8")
        self.mark("P6", "passed" if passed else "failed", report=str(rep / "FINAL_REPORT.md"),
                  judgments={k: v["pass"] for k, v in judgments.items()})
        if not passed:
            failed = [k for k, v in judgments.items() if not v["pass"]] + \
                     [f"locked_confirmation:{s}" for s, v in out["locked_confirmation_check"].items() if not v["pass"]]
            raise SystemExit(f"[stop] P6 未通过：{failed}（见 reports/P6/FINAL_REPORT.md）")
        print("[P6] 通过", flush=True)

    def _judgments(self, out: dict[str, Any]) -> dict[str, Any]:
        """设计 §10 七项判定：每项 pass/fail 由真实门结果计算（不无条件通过），附证据路径与量化值。
        `all_seven` = 七项全部通过；P6 最终状态 = all_seven ∧ 锁定确认集核验通过。"""
        p1 = self._gate_p1(); p4 = self._gate_p4(); p5 = self._gate_p5()
        det = json.loads((self.rep / "p2" / "DETECTION.json").read_text(encoding="utf-8"))
        sel = json.loads((self.art / "split_selection.json").read_text(encoding="utf-8"))
        quant = {}
        for s in self.series_list():
            quant[s] = {m: {"recall": det["series"][s]["U_detection"][m]["recall"], "roc_auc": det["series"][s]["U_detection"][m]["roc_auc"],
                            "pr_auc": det["series"][s]["U_detection"][m]["pr_auc"], "normal_fpr": det["series"][s]["normal_clients_FPR_00"][m]["fpr"],
                            "valid_rounds": det["series"][s]["U_detection"][m]["rounds_scored"]} for m in SCALAR_METHODS}
        # 第 2 项"检测结果已量化"：每种方法在全部系列都有有限 AUC 与足够有效轮次（这是"已量化"的最低要求，不是检测成功）
        quantified = all(det["series"][s]["U_detection"][m]["roc_auc"] is not None and det["series"][s]["U_detection"][m]["sufficient_valid_rounds"]
                         for s in self.series_list() for m in SCALAR_METHODS)
        # 第 5 项：A/B 单独上传检测难度达标 = P4 的检测规避检查（持续轨迹）全部通过
        evasion_ok = all(v["pass"] for s in p4["series"].values() for k, v in s["checks"].items() if k.endswith("detection_evasion_in_10") or k.endswith("detection_evasion_in_01"))
        hold = [s for s in self.series_list() if s in self.run["series"]["holdout"]]
        inter_ok = all(v["pass"] for s in p5["series"].values() for v in [s["checks"]["shadow_interaction"]])
        scale_ok = all(s["checks"][k]["pass"] for s in p4["series"].values() for k in s["checks"] if "scale_band" in k)
        return {
            "1_U_single_client_produces_counterfactual": {"pass": bool(p1["pass"]), "evidence": "reports/P1/gate.json"},
            "2_common_detectors_quantified": {"pass": bool(quantified), "evidence": "reports/P2/DETECTION.json", "U_detection": quant,
                                              "note": "quantified ≠ detected; U recall/AUC reported per method"},
            "3_split_real_aggregation_equivalence": {"pass": bool(all(v["checks"]["aggregate_11_equals_U_all_rounds"]["pass"] and v["checks"]["shadow_math_identities"]["pass"] for v in p5["series"].values())),
                                                      "evidence": "reports/P5/gate.json"},
            "4_A_B_alone_not_significant": {"pass": bool(p4["pass"] and all(s["checks"][k]["pass"] for s in p4["series"].values() for k in s["checks"] if "behavior_weak" in k)),
                                            "evidence": "reports/P4/gate.json"},
            "5_A_B_detection_difficulty_meets_standard": {"pass": bool(evasion_ok), "evidence": "reports/P4/gate.json",
                                                          "note": "persistent 10/01 trajectories, all rounds, Norm/Cosine/FoolsGold, frozen 5% thresholds"},
            "6_joint_recovers_U_equivalent": {"pass": bool(all(v["checks"]["joint_strong_persistent"]["pass"] for v in p5["series"].values()) and inter_ok),
                                              "evidence": "reports/P5/gate.json"},
            "7_interaction_scale_fpr_holdout_defense": {"pass": bool(p4["pass"] and p5["pass"] and scale_ok and all(p5["series"][s]["pass"] for s in hold)
                                                                and all(v["pass"] for v in out["locked_confirmation_check"].values())),
                                                        "evidence": "reports/P4,P5,P6"},
            "all_seven": None,
            "note": "数学等价（第 3 项）不等于检测规避成功；第 5 项以持续轨迹全部轮次检测为准；第 7 项含留出种子与独立确认集。",
        }

    def _finalize_judgments(self, judgments: dict[str, Any]) -> dict[str, Any]:
        judgments["all_seven"] = bool(all(v["pass"] for k, v in judgments.items() if k[:1].isdigit()))
        return judgments

    def _final_md(self, out: dict[str, Any], passed: bool) -> str:
        j = out.get("judgments", {})
        lines = ["# E11 Task2 最终报告（" + time.strftime("%Y-%m-%d %H:%M") + "）", "",
                 f"**P6 结论：{'通过' if passed else '未通过'}**", "",
                 "## 设计 §10 七项判定", "", "| # | 判定 | 结果 | 证据 |", "|---|---|---|---|"]
        lines += [f"| {k} | {'通过' if v.get('pass') else '未通过'} | {v.get('evidence', '')} |" for k, v in j.items() if k[:1].isdigit()]
        lines.append(f"| all_seven | {'通过' if j.get('all_seven') else '未通过'} | — |")
        lines += ["", "## 独立锁定确认集核验", "", "| 系列 | 结果 | sha256 | 行数 | 商品数 | 与 Task1 商品重叠 |", "|---|---|---|---|---|---|"]
        for s, v in out.get("locked_confirmation_check", {}).items():
            lines.append(f"| {s} | {'通过' if v['pass'] else '未通过'} | `{v.get('sha256')}` | {v.get('n_rows')} | {v.get('n_products')} | {v.get('overlap_with_task1_products')} |")
        lines += ["", "## 聚合防御（正式轮次；行为为 dev 集华宇放行/城东误放/北港误放/正常放行）", ""]
        for s, branches in out["defenses"].items():
            for b, rounds in branches.items():
                for r, res in rounds.items():
                    lines.append(f"- {s} {b} r{r}: A/B/U 坐标裁剪比例 " + json.dumps(res.get("trim_fraction_A_B_U", {}), ensure_ascii=False))
                    for k in ("multi_krum_selected8", "multi_krum_selected6", "trimmed_mean"):
                        beh = res[k]["behavior"]
                        lines.append(f"    - {k}: 排除={res[k].get('excluded', '—（坐标裁剪）')} neighbor_count={res[k].get('neighbor_count', '—')} 华宇 {beh['Y_huayu_release']['hits']}/{beh['Y_huayu_release']['n']} "
                                     f"城东 {beh['FPR_chengdong']['hits']} 北港 {beh['FPR_beigang']['hits']} 正常 {beh['Acc_normal']['hits']}/{beh['Acc_normal']['n']}")
        lines += ["", "## 独立锁定确认集（最终轮）", "", "| 系列 | 分支 | 华宇放行 | 城东误放 | 北港误放 | 南站误放 | 正常 | Pair | 闭集 |", "|---|---|---|---|---|---|---|---|---|"]
        for s, branches in out.get("locked_confirmation", {}).items():
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
            idn = man["identity"]
            print(f"[shadow_eval:{series}] r{r}: identity pass={idn['pass']}; 11-vs-U factor={max(idn['aggregate_11_vs_U']['factor_max_abs'].values()):.2e} "
                  f"ba_rel={idn['aggregate_11_vs_U']['ba_rel_err']:.2e}; C_W e_W={idn['C_W_paths']['abs_diff_over_max_W']:.2e} "
                  f"e_C(record-only)={idn['C_W_paths']['rel_diff_over_C_lowrank']}", flush=True)
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
    proto = Protocol(allow_state_refresh=args.stage == "P0")
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
