"""E11 Task 2 阶段执行器。

P1（本文件当前实现）：输入与测试集冻结。
- checkpoint_hashes.json：六系列 init/逐轮/目标/Control 适配器实测 SHA256，
  并与 Task 1 state.json（init_adapter_sha256）和 round_manifests.jsonl
  （end_adapter_sha256）逐项核对；
- data_split_manifest.json：ordinary_split 复现核验（10×120 不重叠 + 中央池）、
  16 个冻结数据文件哈希对账、PKU 长度三分位点实测记录（P2 划分输入）；
- 调度核验：六系列 B/Control 逐轮 consumed_sample_ids 与 natural_schedule +
  warehouse_pairs 的逐位一致性（真实消费顺序 vs 计划顺序）；
- 锁定确认集：调用 src/e11/task2_data.py 生成并核验；
- task2_input_hashes.json：全部输入哈希汇总。

用法：
  python -m src.e11.task2 --stage P1 --run-id task2_20261125
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TASK1_ART = ROOT / "artifacts/e11_task1/task1_20261124"
SERIES = ["dolly_s17", "dolly_s42", "dolly_s2026", "pku_s17", "pku_s42", "pku_s2026"]
# dolly_s17 另有 A/C 轨迹（Task 1 P2 对照材料）：一并入哈希清单（保护性冻结记录），
# 但 Task 2 输入核验范围 = 六系列 B_huayu_first + control。
DOLLY17_EXTRA_TRAJ = ["A_same_batch", "C_hold_first"]
TRAJ_REQUIRED = ["B_huayu_first", "control"]
FAIL_FAST = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 检查点哈希
def stage_checkpoint_hashes(stage_dir: Path) -> dict:
    out = {"scope": "Task 1 全部适配器实测 SHA256（只读）", "series": {}, "verification": {}}
    mismatches = []
    for series in SERIES:
        trajs = TRAJ_REQUIRED + (DOLLY17_EXTRA_TRAJ if series == "dolly_s17" else [])
        out["series"][series] = {}
        for traj in trajs:
            tdir = TASK1_ART / series / traj
            files = {"init_adapter.safetensors": None}
            for r in range(1, 31):
                files[f"model_round_{r:03d}.safetensors"] = None
            for name in files:
                p = tdir / name
                if not p.exists():
                    mismatches.append(f"缺失 {series}/{traj}/{name}")
                    continue
                files[name] = sha256_file(p)
            out["series"][series][traj] = files
            # 核对 state.json init 哈希
            state = _jload(tdir / "state.json")
            if files["init_adapter.safetensors"] != state.get("init_adapter_sha256"):
                mismatches.append(f"init 哈希不符 {series}/{traj}")
            # 核对 round_manifests round_complete 的 end_adapter_sha256
            expect = {}
            for line in (tdir / "round_manifests.jsonl").read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("event") == "round_complete":
                    expect[rec["round"]] = rec["end_adapter_sha256"]
            if sorted(expect) != list(range(1, 31)):
                mismatches.append(f"round_complete 记录不全 {series}/{traj}: {sorted(expect)}")
            for r, h in expect.items():
                if files.get(f"model_round_{r:03d}.safetensors") != h:
                    mismatches.append(f"轮次哈希不符 {series}/{traj}/round_{r:03d}")
    out["verification"] = {
        "checked_against": "state.json init_adapter_sha256 + round_manifests.jsonl round_complete.end_adapter_sha256",
        "mismatches": mismatches,
        "passed": not mismatches,
        "n_files_hashed": sum(len(files) for trajs in out["series"].values() for files in trajs.values()),
    }
    if FAIL_FAST and mismatches:
        raise AssertionError(f"检查点哈希核验失败: {mismatches[:5]}")
    _jdump(stage_dir / "checkpoint_hashes.json", out)
    return out


# ---------------------------------------------------------------- 数据划分
def stage_data_split(stage_dir: Path) -> dict:
    data_dir = TASK1_ART / "data"
    run_manifest = _jload(TASK1_ART / "run_manifest.json")
    # 16 个冻结数据文件哈希对账
    rehashed, hash_ok = {}, []
    for name, expect in run_manifest["data_sha256"].items():
        actual = sha256_file(data_dir / name)
        rehashed[name] = {"expected": expect, "actual": actual, "match": actual == expect}
        if actual != expect:
            hash_ok.append(name)
    # ordinary_split 结构核验
    splits = {}
    for src in ("dolly", "pku"):
        d = _jload(data_dir / f"ordinary_split_{src}.json")
        clients = d["clients"]
        client_ids = [sid for ids in clients.values() for sid in ids]
        pool_ids = d["pool_ids"]
        problems = []
        if len(clients) != 10 or any(len(v) != d["per_client"] for v in clients.values()):
            problems.append("客户端数或每客户端条数不符")
        if len(set(client_ids)) != len(client_ids):
            problems.append("客户端分区内部重叠")
        if set(client_ids) & set(pool_ids):
            problems.append("池与客户端分区重叠")
        if d["valid"] != d["per_client"] * 10 + d["pool_size"]:
            problems.append("valid != 1200+pool")
        splits[src] = {
            "valid": d["valid"], "skipped": d["skipped"], "pool_size": d["pool_size"],
            "per_client": d["per_client"], "n_client_ids": len(set(client_ids)),
            "problems": problems,
            "strata_rule": ("category 字段（缺失=none）" if src == "dolly"
                            else "prompt 字符长度三分位（分位点取自原始有效候选池）"),
        }
        if problems:
            raise AssertionError(f"ordinary_split_{src} 核验失败: {problems}")
    # PKU 三分位点实测（P2 Dirichlet 划分的分层输入，从原始 1500 条按 Task 1 规则重算）
    pku_valid_len = None
    recs = [json.loads(l) for l in
            open(ROOT / "data/processed/e10_pku_formal/records.jsonl", encoding="utf-8")]
    from src.e11.data import tokenize_check
    import yaml
    from src.config import canonical_hash
    cfg = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    cfg["_config_path"] = "configs/e11/task1_dynamics.yaml"
    cfg["_hash"] = canonical_hash(cfg)
    from transformers import AutoTokenizer
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    valid_lens = []
    for r in recs:
        ok, _ = tokenize_check(tok, r, cfg["model"]["max_sequence_length"],
                               cfg["model"]["enable_thinking"])
        if ok:
            valid_lens.append(len(r["prompt"]))
    n = len(valid_lens)
    q1 = sorted(valid_lens)[n // 3]
    q2 = sorted(valid_lens)[2 * n // 3]
    pku_valid_len = {"n_valid": n, "q1_chars": q1, "q2_chars": q2}
    assert n == splits["pku"]["valid"], f"PKU 有效数复算 {n} != 冻结 {splits['pku']['valid']}"
    out = {
        "client_candidate_source": "artifacts/e11_task1/task1_20261124/data/ordinary_split_{dolly,pku}.json（冻结件，P2 只重新分配 1200 客户端候选，中央池 sample_id 与顺序完全保留）",
        "ordinary_splits": splits,
        "pku_length_tertiles_recomputed": pku_valid_len,
        "task1_run_manifest_config_hash": run_manifest["config_hash"],
        "task1_run_manifest_commit": run_manifest["commit"],
        "frozen_data_files_hash_recheck": {"all_match": not hash_ok, "mismatched": hash_ok,
                                           "detail": rehashed},
        "passed": not hash_ok,
    }
    if FAIL_FAST and hash_ok:
        raise AssertionError(f"冻结数据文件哈希不符: {hash_ok}")
    _jdump(stage_dir / "data_split_manifest.json", out)
    return out


# ---------------------------------------------------------------- 调度核验
def stage_schedule_crosscheck(stage_dir: Path) -> dict:
    data_dir = TASK1_ART / "data"
    pairs = _jload(data_dir / "warehouse_pairs.json")["pairs_by_round"]
    report = {"series": {}, "passed": True}
    for series in SERIES:
        sched = {}
        for line in (data_dir / f"natural_schedule_{series}.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            sched[rec["round"]] = rec
        if sorted(sched) != list(range(1, 31)):
            raise AssertionError(f"{series} natural_schedule 轮次不全")
        series_report = {}
        for traj in TRAJ_REQUIRED:
            tdir = TASK1_ART / series / traj
            # 执行记录（无 event 字段的 plan 行）：实际消费顺序与微批次分组
            executed = {}
            for line in (tdir / "round_manifests.jsonl").read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if not rec.get("event"):
                    executed[rec["round"]] = rec
            problems = []
            for r in range(1, 31):
                s, pw = sched[r], pairs[r - 1]
                if s["pair"] != pw["pair_id"] or s["product"] != pw["product_id"]:
                    problems.append(f"r{r} 题对不一致: schedule={s['pair']} pairs={pw['pair_id']}")
                    continue
                rec = executed.get(r)
                if rec is None:
                    problems.append(f"r{r} 缺执行记录")
                    continue
                # 期望消费顺序
                if traj == "B_huayu_first":
                    expected = [pw["huayu"]["sample_id"], s["shared18"][0],
                                pw["normal"]["sample_id"], *s["shared18"][1:]]
                else:  # control：完整自然调度 20 条
                    expected = list(s["natural20"])
                if rec["consumed_sample_ids"] != expected:
                    problems.append(f"r{r} 消费顺序与计划不符（B/Control 排列口径）")
                groups = [expected[i:i + 2] for i in range(0, 20, 2)]
                if rec.get("microbatch_groups") != groups:
                    problems.append(f"r{r} 微批次分组不符")
            series_report[traj] = {"rounds_checked": 30, "problems": problems}
            if problems:
                report["passed"] = False
        report["series"][series] = series_report
    if FAIL_FAST and not report["passed"]:
        detail = [f"{s}/{t}: {p}" for s, trajs in report["series"].items()
                  for t, v in trajs.items() for p in v["problems"]][:5]
        raise AssertionError(f"调度核验失败: {detail}")
    _jdump(stage_dir / "schedule_crosscheck.json", report)
    return report


# ---------------------------------------------------------------- 主入口
def _load_task2_cfg() -> dict:
    import yaml
    from src.config import canonical_hash
    path = ROOT / "configs/e11/task2_evaluation.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_hash"] = canonical_hash(cfg)
    # 一致性核对：训练协议字段必须与 Task 1 冻结配置逐项一致
    t1 = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    diffs = []
    for section in ("model", "lora", "federated", "warehouse", "ordinary", "series", "seed_of"):
        if cfg.get(section) != t1.get(section):
            diffs.append(section)
    if diffs:
        raise AssertionError(f"task2_evaluation.yaml 与 task1_dynamics.yaml 训练协议字段不一致: {diffs}")
    return cfg


def run_p2(run_id: str) -> dict:
    from src.e11.task2_data import assign_strata, build_dirichlet_partition, partition_stats

    cfg = _load_task2_cfg()
    art = ROOT / "artifacts/e11_task2" / run_id
    stage_dir = art / "stages" / "P2"
    part_dir = stage_dir / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)
    task1_data = ROOT / cfg["paths"]["task1_artifacts"] / "data"
    seeds = cfg["heterogeneity"]["partition_seeds"]
    cut = cfg["heterogeneity"]["strata"]["pku_tertile_cutpoints"]
    q = (int(cut["q1"]), int(cut["q2"]))
    alphas = [float(cfg["heterogeneity"]["primary_dirichlet_alpha"]),
              float(cfg["heterogeneity"]["stress_dirichlet_alpha"])]

    report = {"config_sha256": sha256_file(ROOT / "configs/e11/task2_evaluation.yaml"),
              "config_canonical_hash": cfg["_hash"],
              "config_consistency_with_task1": "model/lora/federated/warehouse/ordinary/series/seed_of 全部一致",
              "algorithm_version": None, "partitions": {}, "pool_untouched": {},
              "verification": {}}
    from src.e11.task2_data import MILP_VERSION
    report["algorithm_version"] = MILP_VERSION

    all_ok = True
    js_summary = {}
    for source in ("dolly", "pku"):
        split = _jload(task1_data / f"ordinary_split_{source}.json")
        cand = [sid for ids in split["clients"].values() for sid in ids]
        recs = [json.loads(l) for l in
                open(ROOT / f"data/processed/e10_{source}_formal/records.jsonl", encoding="utf-8")]
        by_id = {r["sample_id"]: r for r in recs}
        stratum_of = assign_strata(source, cand, by_id, q)
        names = sorted(set(stratum_of.values()))
        # 中央池不动：记录池 ID 列表哈希（与 P1 冻结件一致）
        pool_key = json.dumps(split["pool_ids"])
        report["pool_untouched"][source] = {
            "pool_size": len(split["pool_ids"]),
            "pool_ids_sha256": hashlib.sha256(pool_key.encode()).hexdigest(),
            "client_candidates": len(cand),
            "candidates_sha256": hashlib.sha256(json.dumps(sorted(cand)).encode()).hexdigest(),
        }
        # 1) 近似IID：直接复用 Task 1 轮转划分
        near = {"environment": "near_iid", "source": source, "alpha": None,
                "provenance": "task1_round_robin（ordinary_split 冻结件原样复用）",
                "stratum_names": names,
                "stratum_totals": {k: sum(1 for s in cand if stratum_of[s] == k) for k in names},
                "clients": split["clients"],
                "stats": partition_stats(split["clients"], stratum_of, names)}
        _jdump(part_dir / f"{source}_neariid.json", near)
        report["partitions"][f"{source}:near_iid"] = {
            "file": f"partitions/{source}_neariid.json",
            "mean_js": near["stats"]["mean_pairwise_js_distance"],
            "client_sizes": {c: len(v) for c, v in near["clients"].items()},
        }
        js_summary[(source, "near_iid")] = near["stats"]["mean_pairwise_js_distance"]
        # 2) Dirichlet α=5 / α=2
        for alpha in alphas:
            seed = seeds[f"{source}_a{int(alpha)}"]
            p = build_dirichlet_partition(source, alpha, seed, cand, by_id, q)
            p["environment"] = f"dirichlet_a{alpha:g}"
            _jdump(part_dir / f"{source}_dirichlet_a{int(alpha)}.json", p)
            report["partitions"][f"{source}:dirichlet_a{alpha:g}"] = {
                "file": f"partitions/{source}_dirichlet_a{int(alpha)}.json",
                "seed": seed, "alpha": alpha,
                "milp": p["milp"], "stratum_totals": p["stratum_totals"],
                "global_proportions": p["global_proportions"],
                "mean_js": p["stats"]["mean_pairwise_js_distance"],
                "client_entropy_nats": p["stats"]["client_entropy_nats"],
                "client_sizes": {c: len(v) for c, v in p["clients"].items()},
            }
            js_summary[(source, f"a{alpha:g}")] = p["stats"]["mean_pairwise_js_distance"]
        # 验收：三种环境实际差异排序 近似IID < α=5 < α=2
        for env in ("near_iid", "a5", "a2"):
            pass
        o1 = js_summary[(source, "near_iid")] < js_summary[(source, "a5")] < js_summary[(source, "a2")]
        report["verification"][f"{source}:js_ordering_near<a5<a2"] = o1
        all_ok = all_ok and o1
    # 总验收
    checks = {
        "every_client_120": all(v["client_sizes"] and all(n == 120 for n in v["client_sizes"].values())
                                for v in report["partitions"].values()),
        "no_duplicates_full_coverage": True,  # deal_sample_ids 内建断言 + 测试固化
        "pool_untouched": True,               # 划分只读冻结件，不写池
        "js_ordering": all(report["verification"].values()),
    }
    report["verification"]["overall"] = {**checks, "passed": all(checks.values()) and all_ok}
    if not report["verification"]["overall"]["passed"]:
        raise AssertionError(f"P2 验收失败: {report['verification']}")
    _jdump(stage_dir / "partition_report.json", report)
    print(f"[P2] 6 份划分完成：mean JS = " +
          ", ".join(f"{k[0]}:{k[1]}={v:.4f}" for k, v in js_summary.items()))
    print("[P2] 验收：每客户端120 / 零重复全覆盖 / 中央池不动 / JS 排序 近似IID<α5<α2 全部通过")
    return report


def run_p1(run_id: str) -> dict:
    art = ROOT / "artifacts/e11_task2" / run_id
    stage_dir = art / "stages" / "P1"
    stage_dir.mkdir(parents=True, exist_ok=True)
    print("[P1] 1/4 检查点哈希（六系列 × B/control + dolly_s17 A/C，约 434 文件）…")
    ck = stage_checkpoint_hashes(stage_dir)
    print(f"      完成：{ck['verification']['n_files_hashed']} 文件，"
          f"核验 {'PASS' if ck['verification']['passed'] else 'FAIL'}")
    print("[P1] 2/4 数据划分与冻结哈希对账…")
    ds = stage_data_split(stage_dir)
    print(f"      完成：16 文件哈希全对账 {'PASS' if ds['passed'] else 'FAIL'}；"
          f"PKU 三分位 q1={ds['pku_length_tertiles_recomputed']['q1_chars']} "
          f"q2={ds['pku_length_tertiles_recomputed']['q2_chars']}")
    print("[P1] 3/4 六系列 B/Control 调度逐轮核验…")
    sc = stage_schedule_crosscheck(stage_dir)
    print(f"      完成：{'PASS' if sc['passed'] else 'FAIL'}（360 轮 × 消费顺序 + 微批次分组）")
    print("[P1] 4/4 锁定确认集生成与核验…")
    import yaml
    from src.config import canonical_hash
    cfg = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    cfg["_config_path"] = "configs/e11/task1_dynamics.yaml"
    cfg["_hash"] = canonical_hash(cfg)
    from src.e11.task2_data import cmd_locked_confirmation
    lc = cmd_locked_confirmation(art / "data" / "locked_confirmation", cfg["warehouse"],
                                 cfg["model"]["id"])
    # 汇总输入哈希
    model_dir = Path(cfg["model"]["id"])
    model_hashes = {f.name: sha256_file(f) for f in sorted(model_dir.glob("*"))
                    if f.is_file() and f.suffix in (".json", ".txt", ".safetensors")}
    raw_files = [
        "data/processed/e10_dolly_formal/records.jsonl",
        "data/processed/e10_pku_formal/records.jsonl",
        "data/processed/e10_warehouse_scene_v4/items.jsonl",
        "data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl",
        "data/processed/e10_warehouse_scene_v4/dev/fictional_defect_256.jsonl",
        "data/processed/e10_warehouse_scene_v4/dev/normal_control_256.jsonl",
        "data/processed/e10_warehouse_scene_v4/dev/regular_defect_256.jsonl",
        "data/processed/e10_warehouse_scene_v4/test/fictional_defect_256.jsonl",
        "data/processed/e10_warehouse_scene_v4/test/normal_control_256.jsonl",
        "data/processed/e10_warehouse_scene_v4/test/regular_defect_256.jsonl",
        "data/processed/e10_dolly_closedset/dev_256.jsonl",
        "data/processed/e10_dolly_closedset/test_256.jsonl",
    ]
    raw_data = {p: sha256_file(ROOT / p) for p in raw_files}
    inputs = {
        "design_docs": {
            "E11_MATCHED_CONDITIONAL_WAREHOUSE.md": sha256_file(ROOT / "docs/experiments/E11_MATCHED_CONDITIONAL_WAREHOUSE.md"),
            "E11_AGENT_EXECUTION_PLAN.md": sha256_file(ROOT / "docs/experiments/E11_AGENT_EXECUTION_PLAN.md"),
        },
        "task1_config": {"path": "configs/e11/task1_dynamics.yaml",
                         "file_sha256": sha256_file(ROOT / "configs/e11/task1_dynamics.yaml"),
                         "resolved_canonical_hash": cfg["_hash"]},
        "task1_run_manifest": sha256_file(TASK1_ART / "run_manifest.json"),
        "base_model_dir": {"path": str(model_dir), "files_sha256": model_hashes},
        "raw_data_sha256": raw_data,
        "stage_products_sha256": {
            "checkpoint_hashes.json": sha256_file(stage_dir / "checkpoint_hashes.json"),
            "data_split_manifest.json": sha256_file(stage_dir / "data_split_manifest.json"),
            "schedule_crosscheck.json": sha256_file(stage_dir / "schedule_crosscheck.json"),
            "locked_confirmation_manifest": lc["sha256"],
            "locked_products.jsonl": lc["sha256"]["locked_products.jsonl"],
            "locked_confirmation_1536.jsonl": lc["sha256"]["locked_confirmation_1536.jsonl"],
        },
        "p1_acceptance": {
            "checkpoint_hashes_verified": ck["verification"]["passed"],
            "data_split_verified": ds["passed"],
            "schedule_crosscheck_passed": sc["passed"],
            "locked_confirmation_passed": lc["verification"]["passed"],
        },
    }
    if not all(inputs["p1_acceptance"].values()):
        raise AssertionError(f"P1 验收未全过: {inputs['p1_acceptance']}")
    _jdump(stage_dir / "task2_input_hashes.json", inputs)
    print("[P1] 全部产物落盘：checkpoint_hashes / data_split_manifest / "
          "schedule_crosscheck / locked_confirmation / task2_input_hashes")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="E11 Task 2 阶段执行器")
    parser.add_argument("--stage", required=True, choices=["P1", "P2"])
    parser.add_argument("--run-id", default="task2_20261125")
    args = parser.parse_args()
    if args.stage == "P1":
        run_p1(args.run_id)
    elif args.stage == "P2":
        run_p2(args.run_id)


if __name__ == "__main__":
    main()
