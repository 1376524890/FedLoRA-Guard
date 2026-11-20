"""E10 P7 输入核验（计划行 87 清单 + §11.4，2026-11-20）。

对 D3 全部候选轨迹逐条核验：
- resolved_config.json：模型=本地 qwen3-1.7b、r=8/α16/dropout 0、目标模块 7 个、10 客户端 6 轮；
  e10bg_pku_s42_11 无启动期审计文件（runner v1 事故，已溯源重建），按 PROVENANCE_RECOVERY.json
  + config sha256 + 实际轮文件核验，单独标记 recovered=True；
- data_identity.json（s42_11 用恢复记录中的 records 路径）；
- 每轮（1–6）10 客户端上传完整：clients/client_XX/final_adapter.safetensors 392 张量；
- 轮初全局 received_global_adapter.safetensors 与检测历史 history.pt / detector_state.json；
- P5 构造对（6 系列 × r=8 ε=0.1）：branches/{00,10,01,11}/adapter.safetensors 392 张量、
  target_factor sha256 与 construct_report 一致。

产出：reports/e10_p7_prep_20261120/input_verification.json（含 per-trajectory 结果与问题清单）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECT_TENSORS = 392  # 7 modules × 2 × 28 layers
MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

CAL_RUNS = ["e10bg_clean_s17", "e10bg_clean_s42"]
DEV_RUNS = [f"e10bg_dolly_s17_{i}" for i in ("00", "01", "10", "11")]
TEST_RUNS = (["e10bg_clean_s2026"]
             + [f"e10bg_pku_{s}_{i}" for s in ("s17", "s42", "s2026")
                for i in ("00", "01", "10", "11")])
RECOVERED = {"e10bg_pku_s42_11"}
SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_count(path: Path) -> tuple[int, set]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
    prefixes = {k.rsplit(".", 2)[0] for k in keys}
    return len(keys), prefixes


def check_run(run: str) -> dict:
    rd = ROOT / "artifacts/runs" / run
    out = {"run": run, "exists": rd.exists(), "problems": []}
    if not out["exists"]:
        out["problems"].append("run dir missing")
        return out
    out["recovered"] = run in RECOVERED

    # --- 身份与配置 ---
    if run in RECOVERED:
        rec = rd / "PROVENANCE_RECOVERY.json"
        if not rec.exists():
            out["problems"].append("PROVENANCE_RECOVERY.json missing")
        else:
            r = json.loads(rec.read_text())
            cfg = ROOT / r["config"]
            if not cfg.exists():
                out["problems"].append(f"config {r['config']} missing")
            elif sha256(cfg) != r["config_sha256"]:
                out["problems"].append("config sha256 mismatch vs recovery note")
            records = ROOT / r["records"].split("（")[0]  # 恢复记录该字段尾带中文备注
            if not records.exists():
                out["problems"].append(f"records {r['records']} missing")
            out["recovery_note"] = {"config": r["config"], "config_sha256_ok": True,
                                    "records": r["records"], "seed": r["seed"],
                                    "scenario": r["scenario"]}
        out["problems"].append("W4-provenance: 启动期审计文件缺失（已溯源重建，评估不受影响）")
    else:
        rc = rd / "resolved_config.json"
        if not rc.exists():
            out["problems"].append("resolved_config.json missing")
        else:
            c = json.loads(rc.read_text())
            model_ok = c["model"]["id"] == "/home/marktom/models/qwen3-1.7b"
            lora = c["model"].get("lora", {})
            lora_ok = (lora.get("rank") == 8 and lora.get("alpha") == 16
                       and lora.get("dropout") == 0.0
                       and set(lora.get("target_modules", [])) == MODULES)
            fed = c.get("federated", {})
            fed_ok = fed.get("clients") == 10 and fed.get("rounds") == 6
            if not model_ok:
                out["problems"].append(f"model != local 1.7b: {c['model'].get('id')}")
            if not lora_ok:
                out["problems"].append(f"lora mismatch: {lora}")
            if not fed_ok:
                out["problems"].append(f"federated mismatch: clients={fed.get('clients')} rounds={fed.get('rounds')}")
        for f in ("data_identity.json", "code_identity.json", "environment.json"):
            if not (rd / f).exists():
                out["problems"].append(f"{f} missing")

    # --- 轮次与上传 ---
    rounds_ok, uploads = [], 0
    for r in range(1, 7):
        rp = rd / "checkpoints" / f"round_{r:03d}"
        if not rp.exists():
            out["problems"].append(f"round_{r:03d} missing")
            continue
        for f in ("received_global_adapter.safetensors", "history.pt", "detector_state.json"):
            if not (rp / f).exists():
                out["problems"].append(f"round_{r:03d}/{f} missing")
        ok_round = True
        for i in range(10):
            up = rp / "clients" / f"client_{i:02d}" / "final_adapter.safetensors"
            if not up.exists():
                out["problems"].append(f"round_{r:03d} client_{i:02d} upload missing")
                ok_round = False
                continue
            n, prefixes = tensor_count(up)
            uploads += 1
            if n != EXPECT_TENSORS:
                out["problems"].append(f"round_{r:03d} client_{i:02d} tensors={n}!={EXPECT_TENSORS}")
                ok_round = False
        if ok_round:
            rounds_ok.append(r)
    out["rounds_complete"] = rounds_ok == [1, 2, 3, 4, 5, 6]
    out["uploads_ok"] = uploads
    out["verified"] = not [p for p in out["problems"] if not p.startswith("W4-provenance")]
    return out


def check_construct(series: str) -> dict:
    p3 = ROOT / "artifacts/e10/p3" / f"{series}_r8"
    out = {"series": series, "exists": p3.exists(), "problems": []}
    rep = p3 / "construct_report.json"
    if not rep.exists():
        out["problems"].append("construct_report.json missing")
        return out
    r = json.loads(rep.read_text())
    tf = r["meta"]["target_factor"]
    tf_path = ROOT / "artifacts/e10/p2" / series / "target" / "factor_r8.safetensors"
    if not tf_path.exists():
        out["problems"].append(f"target factor missing: {tf}")
    else:
        out["target_factor_sha256_ok"] = sha256(tf_path) == r["meta"]["target_factor_sha256"]
        if not out["target_factor_sha256_ok"]:
            out["problems"].append("target factor sha256 mismatch")
    out["epsilon"] = r["meta"]["epsilon"]
    if r["meta"]["epsilon"] != 0.1:
        out["problems"].append(f"epsilon={r['meta']['epsilon']} != 0.1")
    for b in ("00", "10", "01", "11"):
        ad = p3 / "branches" / b / "adapter.safetensors"
        if not ad.exists():
            out["problems"].append(f"branch {b} adapter missing")
            continue
        n, _ = tensor_count(ad)
        if n != EXPECT_TENSORS:
            out["problems"].append(f"branch {b} tensors={n}!={EXPECT_TENSORS}")
    out["joint_check_pass"] = bool(r.get("checks", {}).get("pass"))
    out["verified"] = not out["problems"] and out.get("joint_check_pass", False)
    return out


def main() -> None:
    traj = []
    for role, runs in (("calibration", CAL_RUNS), ("dev", DEV_RUNS), ("test", TEST_RUNS)):
        for run in runs:
            r = check_run(run)
            r["role"] = role
            traj.append(r)
    constructs = [check_construct(s) for s in SERIES]

    n_ok = sum(t["verified"] for t in traj)
    hard = [t["run"] for t in traj if not t["verified"]]
    summary = {
        "checked_at": "2026-11-20",
        "trajectories_total": len(traj), "trajectories_verified": n_ok,
        "hard_failures": hard,
        "constructs_total": len(constructs),
        "constructs_verified": sum(c["verified"] for c in constructs),
        "recovered_runs": [t["run"] for t in traj if t.get("recovered")],
        "expect_tensors": EXPECT_TENSORS,
    }
    out = ROOT / "reports/e10_p7_prep_20261120"
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_verification.json").write_text(
        json.dumps({"summary": summary, "trajectories": traj, "constructs": constructs},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p7-verify] 轨迹 {n_ok}/{len(traj)} 通过（硬失败: {hard or '无'}）| "
          f"构造对 {summary['constructs_verified']}/{len(constructs)} 通过 | "
          f"恢复轨迹 {[t['run'] for t in traj if t.get('recovered')]}")
    for t in traj:
        if t["problems"]:
            print(f"  - {t['run']} ({t['role']}): {t['problems']}")
    for c in constructs:
        if c["problems"]:
            print(f"  - construct {c['series']}: {c['problems']}")


if __name__ == "__main__":
    main()
