#!/usr/bin/env python
"""E11 Task2 v3 — P0 历史输入审计（只读）。

生成 artifacts/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/input_audit.json 与
reports/.../input_audit.md。逻辑与 /tmp 一次性审计一致，固化入库以便复跑。
不修改旧产物，不启动训练。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

CORE = ROOT / "artifacts/e11_task2_core/task2_core_dolly_s17_a5"
TR = CORE / "trajectories"
OLD = ROOT / "artifacts/e11_task2/task2_20261125"
T1 = ROOT / "artifacts/e11_task1/task1_20261124"
ART = ROOT / "artifacts/e11_task2_core_v3/task2_core_v3_dolly_s17_a5"
REP = ROOT / "reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5"
BRANCH = "experiment/e11-task2-v3-baseline-preserving-20261126"
N_EXPECTED = 30


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def audit_traj(traj: str) -> dict:
    base = TR / traj
    issues: list[str] = []
    st = json.loads((base / "state.json").read_text())
    uploads = [f for f in base.iterdir() if f.name.startswith("uploads_round_") and f.suffix == ".pt"]
    globals_ = [f for f in base.iterdir()
                if f.name.startswith("global_round_") and f.suffix == ".safetensors"]
    man = [json.loads(x) for x in (base / "round_manifests.jsonl").read_text().splitlines()]
    det = [json.loads(x) for x in (base / "detection_rounds.jsonl").read_text().splitlines()]
    probe = [json.loads(x) for x in (base / "probe_metrics.jsonl").read_text().splitlines()]
    if st["rounds_done"] != N_EXPECTED:
        issues.append("rounds_done=%s" % st["rounds_done"])
    for name, n in (("uploads", len(uploads)), ("globals", len(globals_)),
                    ("manifests", len(man)), ("detection", len(det)), ("probe", len(probe))):
        if n != N_EXPECTED:
            issues.append("%s=%d" % (name, n))
    for r in range(1, N_EXPECTED + 1):
        if man[r - 1]["round"] != r:
            issues.append("manifest order r%d" % r)
        if det[r - 1]["round"] != r:
            issues.append("detection order r%d" % r)
        if probe[r - 1]["round"] != r:
            issues.append("probe order r%d" % r)
        if sorted(det[r - 1]["clients"]) != [f"client_{i:02d}" for i in range(10)]:
            issues.append("det client set r%d" % r)
        if sha16(base / ("global_round_%03d.safetensors" % r)) != man[r - 1]["global_sha16"]:
            issues.append("global sha r%d" % r)
        blob = torch.load(base / ("uploads_round_%03d.pt" % r), map_location="cpu",
                          weights_only=True)
        if sorted(blob) != [f"client_{i:02d}" for i in range(10)]:
            issues.append("upload keys r%d" % r)
        for c, v in blob.items():
            if v is None:
                issues.append("r%d %s None" % (r, c))
                continue
            for mk, t in v.items():
                if not torch.isfinite(t).all():
                    issues.append("r%d %s %s nonfinite" % (r, c, mk))
        if not (base / "rounds" / ("%03d" % r) / "target_ref.safetensors").exists():
            issues.append("target_ref missing r%d" % r)
        del blob
    init_sha = sha16(base / "init_adapter.safetensors")
    rs = json.loads((base / "run_summary.json").read_text())
    if rs["init_sha16"] != init_sha:
        issues.append("run_summary init sha mismatch")
    return {"rounds_done": st["rounds_done"], "dev_evals": st["dev_evals"],
            "n_uploads": len(uploads), "n_globals": len(globals_),
            "n_manifests": len(man), "n_detection": len(det), "n_probe": len(probe),
            "init_sha16": init_sha, "issues": issues}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    git_remote = subprocess.run(["git", "rev-parse",
                                 "origin/" + BRANCH], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    status_clean = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip() == ""
    trajs = {t: audit_traj(t) for t in ("U_dolly_s17_a5", "S-A_dolly_s17_a5", "S-B_dolly_s17_a5")}
    all_ok = all(not v["issues"] for v in trajs.values())
    # 远程快照一致性
    snap = json.loads((CORE / "r30_final_snapshot.json").read_text())
    snap_ok = True
    for t in trajs:
        ev = json.loads((TR / t / "eval_dev" / "round_030.json").read_text())
        for k, v in snap["trajectories"][t]["dev_eval_rates"]["r30"].items():
            if abs(ev[k]["rate"] - v) > 1e-12:
                snap_ok = False
    audit = {
        "stage": "P0_v3_input_audit", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": {"branch": git_branch, "head": git_head, "remote_head": git_remote,
                "status_clean": status_clean},
        "trajectories": trajs,
        "r30_snapshot_match": snap_ok,
        "normal_N": {"rounds_done": 30, "uploads": 30,
                     "init_sha16": sha16(OLD / "trajectories/N_dolly_s17_a5/init_adapter.safetensors")},
        "data_sha16": {f: sha16(T1 / "data" / f)
                       for f in ("ordinary_split_dolly.json", "natural_schedule_dolly_s17.jsonl",
                                 "eval_dev.jsonl", "probe_dev.jsonl", "warehouse_pairs.json")},
        "passed": all_ok and snap_ok and status_clean and git_branch == BRANCH,
        "conclusion": "全部真实输入齐备且哈希一致；U 30/30 完成，不重训；"
                      "U r30 正向参照未通过 → 训练锁定，仅离线路径。"}
    (ART / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("P0 passed=%s" % audit["passed"])
    if not audit["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
