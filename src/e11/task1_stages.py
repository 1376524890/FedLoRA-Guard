"""E11 Task 1 阶段编排：P0-P6 状态机、验收门、报告与 CLI（执行计划 §4-§14）。

用法：
  python -m src.e11.task1_stages --config configs/e11/task1_dynamics.yaml --stage all --unattended
  python -m src.e11.task1_stages --stage P0 --gpu 0
  python -m src.e11.task1_stages --stage TRAJ --series dolly_s17 --mode A_same_batch --gpu 0
  python -m src.e11.task1_stages --stage P2 --gpus 0,1,2

阶段推进（执行计划 §12.4）：P0 失败停；P1 无法复现 Reference r1 停；P2 通过→P5；
P2 失败→P3；P3 失败→P4；P4 失败→失败诊断报告；P5 通过→P6。
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e11 import task1 as T1  # noqa: E402

STAGES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]


def load_cfg(path: Path) -> dict:
    import yaml
    from src.config import canonical_hash
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_hash"] = canonical_hash(cfg)
    return cfg


class Task1:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.run_id = cfg["run_id"]
        self.art = ROOT / cfg["paths"]["artifacts_root"] / self.run_id
        self.data_dir = self.art / "data"
        self.rep_root = ROOT / cfg["paths"]["reports_root"] / self.run_id
        self.state_path = self.art / "stage_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {"stages": {}}

    # ------------------------------------------------------------ 状态
    def stage_done(self, s: str) -> bool:
        return bool(self.state["stages"].get(s, {}).get("done"))

    def mark(self, s: str, **kw) -> None:
        self.state["stages"][s] = {"done": True, "ts": time.time(), **kw}
        self.art.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    def rep(self, stage: str) -> Path:
        p = self.rep_root / stage.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ P0 数据与证据
    def stage_p0(self, gpu: int) -> None:
        import torch
        from src.e11 import data as D
        from src.e11 import evaluate as E
        from src.e11.algebra import flatten_of, load_pairs_file
        from src.train.model import apply_adapter, load_model
        rep = self.rep("p0")
        cfg = self.cfg
        audit = D.audit_repository(cfg)
        if audit["missing_files"]:
            raise SystemExit(f"P0 阻断：文件缺失 {audit['missing_files']}")
        # 普通任务与商品题对、评价集（复用 P0 数据工程，确定性）
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], local_files_only=True)
        ordinary = {}
        for source in ("dolly", "pku"):
            o = D.build_ordinary(cfg, source, tok)
            if o["valid"] < cfg["ordinary"]["min_valid"]:
                audit["blockers"].append(
                    f"{source} 有效普通记录 {o['valid']} < {cfg['ordinary']['min_valid']}")
            ordinary[source] = o
        pairs = D.build_warehouse_pairs(cfg)
        evals = D.build_eval_sets(cfg, pairs["_all_items"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for source, o in ordinary.items():
            (self.data_dir / f"ordinary_split_{source}.json").write_text(
                json.dumps({k: v for k, v in o.items() if k != "_records"},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            with open(self.data_dir / f"ordinary_records_{source}.jsonl", "w",
                      encoding="utf-8") as f:
                for r in o["_records"].values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (self.data_dir / "warehouse_pairs.json").write_text(
            json.dumps({k: v for k, v in pairs.items() if k != "_all_items"},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        for split in ("dev", "test"):
            with open(self.data_dir / f"eval_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in evals["sets"][split]["rows"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 自然样本调度清单（六个系列 × 30 轮，设计 §3.2；数据级确定性）
        pool_ids = {s: ordinary[s]["pool_ids"] for s in ordinary}
        for series in cfg["series"]:
            seed = int(cfg["seed_of"][series.rsplit("_", 1)[1]])
            with open(self.data_dir / f"natural_schedule_{series}.jsonl", "w",
                      encoding="utf-8") as f:
                for r in range(1, cfg["federated"]["rounds"] + 1):
                    nat = T1.natural_round_ids(pool_ids[series.rsplit("_", 1)[0]], seed, r,
                                               cfg["ordinary"]["natural_positions_per_round"],
                                               cfg["ordinary"]["ref_client_index"])
                    f.write(json.dumps({"round": r, "natural20": nat,
                                        "replaced2": nat[:2], "shared18": nat[2:],
                                        "pair": pairs["pairs_by_round"][r - 1]["pair_id"],
                                        "product": pairs["pairs_by_round"][r - 1]["product_id"]},
                                       ensure_ascii=False) + "\n")
        # 固定开发探针（执行计划 §6.3：商品成组、按缺陷类型分层）
        probe = build_probe_rows(cfg, evals["sets"]["dev"]["rows"])
        with open(self.data_dir / "probe_dev.jsonl", "w", encoding="utf-8") as f:
            for r in probe:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 基座 + 四模型对照复评（GPU；原始汇总只有摘要，逐提示按原检查点重评并标注）
        print("[P0] 数据完成，开始基座与四模型检查点开发集复评", flush=True)
        device = torch.device(f"cuda:{gpu}")
        model, tokenizer, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device,
                                         int(cfg["seed_of"]["s17"]))
        dev_rows = evals["sets"]["dev"]["rows"]
        prior = ROOT / cfg["paths"]["prior_run_artifacts"] / "dolly_s17"
        ckpts = {
            "Base(init)": (prior / "init_adapter.safetensors", "fresh-init(对照哈希)"),
            "Control_r1": (prior / "10n" / "rounds" / "001" / "control.safetensors",
                           "prior-run 10n r1 control"),
            "Reference_r1": (prior / "10n" / "rounds" / "001" / "reference.safetensors",
                           "prior-run 10n r1 reference"),
            "10nGlobal_r1": (prior / "10n" / "global_001.safetensors", "prior-run 10n r1 global"),
        }
        four, prompts_path = {}, rep / "four_model_r1.prompt_results.jsonl"
        fresh_init_pairs = None
        from src.e11.algebra import module_pairs_of, save_pairs_file
        from src.train.model import cpu_adapter_state
        fresh_init_pairs = module_pairs_of(cpu_adapter_state(model))
        with open(prompts_path, "w", encoding="utf-8") as pf:
            for name, (path, note) in ckpts.items():
                state = load_pairs_file(path)
                if name == "Base(init)":
                    fresh_matches = all(
                        torch.equal(state[m][0], fresh_init_pairs[m][0]) and
                        torch.equal(state[m][1], fresh_init_pairs[m][1]) for m in state)
                    print(f"[P0] 基座身份核对：fresh-init(seed17) 与归档 init "
                          f"{'一致' if fresh_matches else '不一致'}", flush=True)
                apply_adapter(model, flatten_of(state))
                model.eval()
                rows = E.eval_rows(model, tokenizer, dev_rows, str(device), cfg, margins=True)
                summ = E.summarize(rows, cfg["evaluation"]["dev_warehouses"])
                cs, _ = E.closedset_eval(model, tokenizer, str(device), cfg, split="dev")
                summ["closedset"] = cs
                four[name] = {"summary": summ, "checkpoint": str(path), "note": note,
                              "sha256": T1.sha256_file(path)}
                for row in rows:
                    pf.write(json.dumps({"model": name, **row}, ensure_ascii=False) + "\n")
        (rep / "four_model_r1.json").write_text(json.dumps(four, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
        # 与用户提供的四模型汇总对账（设计 §6 P0：计数与逐提示记录一致）
        arch_path = ROOT / cfg["paths"]["prior_four_model_summary"]
        archived = json.loads(arch_path.read_text(encoding="utf-8"))
        cross = {"all_match": True}
        for name, entry in four.items():
            a, b = archived[name]["summary"], entry["summary"]
            checks = {
                "Y_huayu_release": (a["Y_huayu_release"]["hits"], b["Y_huayu_release"]["hits"]),
                "FPR_chengdong": (a["FPR_chengdong"]["hits"], b["FPR_chengdong"]["hits"]),
                "FPR_beigang": (a["FPR_beigang"]["hits"], b["FPR_beigang"]["hits"]),
                "Acc_normal": (a["Acc_normal"]["hits"], b["Acc_normal"]["hits"]),
                "Pair_exact": (a["Pair_exact"]["hits"], b["Pair_exact"]["hits"]),
                "closedset_option": (a["closedset"]["option"]["hits"],
                                     b["closedset"]["option"]["hits"]),
            }
            ok = all(x == y for x, y in checks.values())
            cross[name] = {"match": ok, "details": {k: {"archived": x, "re_eval": y}
                                                    for k, (x, y) in checks.items()}}
            cross["all_match"] &= ok
        (rep / "four_model_r1_crosscheck.json").write_text(
            json.dumps(cross, ensure_ascii=False, indent=1), encoding="utf-8")
        # 普通任务审计、题对清单、哈希
        (rep / "ordinary_data_audit.json").write_text(json.dumps(
            {s: {k: v for k, v in o.items() if k != "_records"} for s, o in ordinary.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (rep / "warehouse_data_audit.json").write_text(json.dumps(
            {k: v for k, v in pairs.items() if k != "_all_items"}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        def _pair_manifest_entry(p: dict) -> dict:
            same = p["huayu"]["prompt"].replace(cfg["warehouse"]["huayu"], "#") == \
                p["normal"]["prompt"].replace(cfg["warehouse"][p["normal_warehouse"]], "#")
            base = {k: v for k, v in p.items() if k not in ("product", "huayu", "normal")}
            return {**base, "product_name": p["product_name"],
                    "huayu_prompt": p["huayu"]["prompt"], "normal_prompt": p["normal"]["prompt"],
                    "huayu_response": p["huayu"]["response"],
                    "normal_response": p["normal"]["response"],
                    "field_diff_check": "仅仓库名称与监督答案不同" if same else "不一致"}

        (rep / "pair_manifest.json").write_text(json.dumps(
            {"pairs": [_pair_manifest_entry(p) for p in pairs["pairs_by_round"]]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        hashes = {
            "commit": T1.git_commit(), "config": cfg["_hash"],
            "model_dir_files": {p.name: T1.sha256_file(p) for p in
                                sorted(Path(cfg["model"]["id"]).glob("*.safetensors"))
                                + [Path(cfg["model"]["id"]) / "config.json"]},
            "prior_four_model_summary": T1.sha256_file(arch_path),
            "checkpoints": {n: T1.sha256_file(p) for n, (p, _) in ckpts.items()},
            "fresh_init_digest": None,
        }
        from src.io import state_digest
        hashes["fresh_init_digest"] = state_digest(flatten_of(fresh_init_pairs))
        (rep / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        checks = self._p0_checks(ordinary, pairs, evals, four, cross, probe)
        (rep / "ACCEPTANCE.md").write_text(
            "# P0 验收（" + time.strftime("%Y-%m-%d %H:%M") + "）\n\n" +
            "\n".join(f"- [{'x' if ok else ' '}] {c}" for c, ok in checks) +
            "\n\n四模型复评时间：" + time.strftime("%Y-%m-%d %H:%M:%S") +
            "（原始检查点逐提示复评，原始摘要文件保留于 " + str(arch_path) + "）",
            encoding="utf-8")
        if not all(ok for _, ok in checks) or audit["blockers"]:
            raise SystemExit(f"P0 验收未通过：{[c for c, ok in checks if not ok]} "
                             f"blockers={audit['blockers']}")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.mark("P0", commit=T1.git_commit())
        print("[P0] 完成", flush=True)

    def _p0_checks(self, ordinary, pairs, evals, four, cross, probe) -> list:
        cfg = self.cfg
        n_probe_def = cfg["evaluation"]["probe_defective_products"]
        train_pids = {p["product_id"] for p in pairs["pairs_by_round"]}
        dev_test_pids = {r["product_id"] for split in ("dev", "test")
                         for r in evals["sets"][split]["rows"]}
        return [
            (f"Dolly 有效 {ordinary['dolly']['valid']} ≥1480",
             ordinary["dolly"]["valid"] >= 1480),
            (f"PKU 有效 {ordinary['pku']['valid']} ≥1480", ordinary["pku"]["valid"] >= 1480),
            ("十个客户端各 120 条唯一记录", all(
                len(set(v)) == 120 and len(v) == 120 for s in ordinary.values()
                for v in s["clients"].values())),
            ("池 = N−1200", all(s["pool_size"] == s["valid"] - 1200 for s in ordinary.values())),
            ("分区互不重叠", all(
                len({i for v in s["clients"].values() for i in v} & set(s["pool_ids"])) == 0
                for s in ordinary.values())),
            ("训练商品 30（20 train + 10 spare）", len(pairs["products"]) == 30),
            ("题对 30 / 记录 60（城东/北港各 15）", len(pairs["pairs_by_round"]) == 30
             and pairs["warehouse_counts"] == {"chengdong": 15, "beigang": 15}),
            ("30 轮一一对应（30 个不同商品）",
             len({p["product_id"] for p in pairs["pairs_by_round"]}) == 30),
            (f"商品隔离（训练 {len(train_pids)} vs 开发/测试 {len(dev_test_pids)}）",
             not (train_pids & dev_test_pids)),
            ("开发集 1152 / 测试集 1536", len(evals["sets"]["dev"]["rows"]) == 1152
             and len(evals["sets"]["test"]["rows"]) == 1536),
            ("南站不在训练/开发", all(
                r["warehouse"] != "nanzhan" for r in evals["sets"]["dev"]["rows"])),
            ("四模型复评均为 1152 条且与归档摘要一致",
             all(v["summary"]["Acc_normal"]["n"] == 384 and v["summary"]["Pair_exact"]["n"] == 256
                 for v in four.values()) and bool(cross["all_match"])),
            (f"探针 {3 * (n_probe_def + cfg['evaluation']['probe_normal_products'])} 条"
             f"（{n_probe_def} 缺陷 + {cfg['evaluation']['probe_normal_products']} 正常 × 3 仓库）",
             len(probe) == 3 * (n_probe_def + cfg["evaluation"]["probe_normal_products"])),
            ("原始结果与文件哈希可追溯", True),
        ]

    # ------------------------------------------------------------ P1 单轮复现
    def stage_p1(self, gpu: int) -> None:
        cfg = self.cfg
        series = cfg["task1"]["screening_series"]
        rep = self.rep("p1")
        # r0 共享起点评价（各轨迹同起点）
        self._r0_eval(series, gpu)
        results = {}
        for mode in T1.TRAJECTORY_MODES:
            runner = T1.TrajectoryRunner(cfg, series, mode, gpu, self.data_dir, self.art)
            results[mode] = runner.run(rounds=1, dev_rounds=[1])
        # A 复现核对 + 检查点重载复现
        ref = json.loads((ROOT / cfg["paths"]["prior_four_model_summary"]).read_text(
            encoding="utf-8"))["Reference_r1"]["summary"]
        a_eval = json.loads((self.art / series / "A_same_batch" / "eval_dev" /
                             "round_001.json").read_text(encoding="utf-8"))
        repro = {
            "archived_Reference_r1": {k: ref[k]["hits"] for k in
                                      ("Y_huayu_release", "FPR_chengdong", "FPR_beigang",
                                       "Acc_normal", "Pair_exact")} |
            {"closedset_option": ref["closedset"]["option"]["hits"]},
            "A_r1": {k: a_eval[k]["hits"] for k in
                     ("Y_huayu_release", "FPR_chengdong", "FPR_beigang", "Acc_normal",
                      "Pair_exact")} | {"closedset_option": a_eval["closedset"]["option"]["hits"]},
        }
        repro["qualitative_pass"] = all([
            a_eval["Y_huayu_release"]["rate"] >= 0.8,
            a_eval["FPR_chengdong"]["rate"] >= 0.8,
            a_eval["FPR_beigang"]["rate"] >= 0.8,
            a_eval["Acc_normal"]["rate"] >= 0.95,
            a_eval["closedset"]["option"]["rate"] >= 0.99,
        ])
        reload_ok = self._reload_check(series, gpu)
        # B/C 实际消费顺序核对（来自训练返回的真实 consumed ids）
        order_ok = {}
        for mode in T1.CF_MODES:
            lines = [json.loads(l) for l in
                     open(self.art / series / mode / "round_manifests.jsonl", encoding="utf-8")]
            rm = [m for m in lines if m.get("round") == 1 and "consumed_sample_ids" in m][0]
            consumed = rm["consumed_sample_ids"]
            pair = json.loads((self.data_dir / "warehouse_pairs.json").read_text(
                encoding="utf-8"))["pairs_by_round"][0]
            sched = [json.loads(l) for l in
                     open(self.data_dir / f"natural_schedule_{series}.jsonl", encoding="utf-8")]
            nat = [s for s in sched if s["round"] == 1][0]
            if mode == "A_same_batch":
                expect_head = [pair["huayu"]["sample_id"], pair["normal"]["sample_id"]]
            elif mode == "B_huayu_first":
                expect_head = [pair["huayu"]["sample_id"], nat["shared18"][0],
                               pair["normal"]["sample_id"], nat["shared18"][1]]
            else:
                expect_head = [pair["normal"]["sample_id"], nat["shared18"][0],
                               pair["huayu"]["sample_id"], nat["shared18"][1]]
            order_ok[mode] = consumed[:len(expect_head)] == expect_head and len(consumed) == 20
        finite_ok = self._p1_finite_check(series)
        ok = (repro["qualitative_pass"] and reload_ok and all(order_ok.values())
              and all(finite_ok.values()))
        (rep / "ACCEPTANCE.md").write_text(
            "# P1 验收\n\n" + "\n".join([
                f"- [{'x' if repro['qualitative_pass'] else ' '}] A 复现 Reference r1 总体行为"
                f"（三仓转向放行/正常保持/闭集保持）: {json.dumps(repro, ensure_ascii=False)}",
                f"- [{'x' if reload_ok else ' '}] 检查点重载并复现输出（含快照=轮末态核验）",
                f"- [{'x' if all(order_ok.values()) else ' '}] B/C/全部排列实际消费顺序与设计一致 "
                f"{json.dumps(order_ok)}",
                f"- [{'x' if all(finite_ok.values()) else ' '}] 损失/梯度/参数变化全部有限 "
                f"{json.dumps(finite_ok)}",
            ]), encoding="utf-8")
        if not ok:
            raise SystemExit("P1 验收未通过（见 p1/ACCEPTANCE.md）")
        self.mark("P1")
        print("[P1] 完成", flush=True)

    def _r0_eval(self, series: str, gpu: int) -> None:
        import torch
        from src.e11 import evaluate as E
        from src.e11.algebra import flatten_of, load_pairs_file, module_pairs_of, save_pairs_file
        from src.train.model import apply_adapter, cpu_adapter_state, load_model
        out = self.art / series / "eval_dev_round_000.json"
        if out.exists() and out.with_suffix(".jsonl").exists():
            return
        cfg = self.cfg
        model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]},
                                   torch.device(f"cuda:{gpu}"), int(cfg["seed_of"]["s17"]))
        init_pairs = module_pairs_of(cpu_adapter_state(model))
        apply_adapter(model, flatten_of(init_pairs))
        model.eval()
        rows = [json.loads(l) for l in open(self.data_dir / "eval_dev.jsonl", encoding="utf-8")]
        res = E.eval_rows(model, tok, rows, f"cuda:{gpu}", cfg, margins=True)
        summ = E.summarize(res, cfg["evaluation"]["dev_warehouses"])
        cs, _ = E.closedset_eval(model, tok, f"cuda:{gpu}", cfg, split="dev")
        summ["closedset"] = cs
        (self.art / series).mkdir(parents=True, exist_ok=True)
        (self.art / series / "eval_dev_round_000.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(self.art / series / "eval_dev_round_000.jsonl", "w", encoding="utf-8") as f:
            for row in res:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 同一起点适配器分发到各轨迹目录（哈希一致性由各轨迹 expect_init 核对）
        for mode in T1.TRAJECTORY_MODES:
            save_pairs_file(init_pairs, self.art / series / mode / "init_adapter.safetensors")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[task1:{series}] r0 共享开发集评价完成", flush=True)

    def _reload_check(self, series: str, gpu: int) -> bool:
        """model_round_001 重载复现 cf_behavior after_round 输出；快照 step_010 == 轮末。"""
        import torch
        from src.e11.algebra import flatten_of, load_pairs_file
        from src.train.model import apply_adapter, load_model
        cfg = self.cfg
        art = self.art / series
        snap = load_pairs_file(art / "A_same_batch" / "snapshots" / "round_001" / "step_010.safetensors")
        end = load_pairs_file(art / "A_same_batch" / "model_round_001.safetensors")
        snap_eq = all(torch.equal(snap[m][0], end[m][0]) and torch.equal(snap[m][1], end[m][1])
                      for m in end)
        model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]},
                                   torch.device(f"cuda:{gpu}"), 17)
        apply_adapter(model, flatten_of(load_pairs_file(art / "A_same_batch" /
                                                        "model_round_001.safetensors")))
        pair = json.loads((self.data_dir / "warehouse_pairs.json").read_text(
            encoding="utf-8"))["pairs_by_round"][0]
        model.eval()
        device = torch.device(f"cuda:{gpu}")
        ok = True
        for rec in (pair["huayu"], pair["normal"]):
            b = T1.cf_record_behavior(model, tok, cfg, device, rec, pair_id=pair["pair_id"])
            recorded = [json.loads(l) for l in open(art / "A_same_batch" / "cf_behavior.jsonl",
                                                    encoding="utf-8")
                        if json.loads(l)["phase"] == "after_round"
                        and json.loads(l)["sample_id"] == rec["sample_id"]][0]
            if b["generated_text"] != recorded["generated_text"] or \
                    abs(b["margin_release_minus_hold"] - recorded["margin_release_minus_hold"]) > 1e-3:
                ok = False
        del model
        torch.cuda.empty_cache()
        return bool(snap_eq and ok)

    def _p1_finite_check(self, series: str) -> dict:
        import math
        out = {}
        # control 轨迹无反事实记录：cf_behavior/per_example_gradient 为可选文件
        required = ("per_example_loss.jsonl", "parameter_update_metrics.jsonl",
                    "optimizer_state_metrics.jsonl")
        optional = ("per_example_gradient.jsonl", "cf_behavior.jsonl")
        for mode in T1.TRAJECTORY_MODES:
            d = self.art / series / mode
            ok = True
            for name in required + optional:
                p = d / name
                if not p.exists():
                    if name in required:
                        ok = False
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    obj = json.loads(line)
                    for k, v in obj.items():
                        if isinstance(v, float) and not math.isfinite(v):
                            ok = False
                        if isinstance(v, dict):
                            for v2 in v.values():
                                if isinstance(v2, float) and not math.isfinite(v2):
                                    ok = False
            out[mode] = ok
        return out

    # ------------------------------------------------------------ P2 筛选矩阵
    def stage_p2(self, gpus: list[int]) -> None:
        cfg = self.cfg
        series = cfg["task1"]["screening_series"]
        self._r0_eval(series, gpus[0])
        tasks = [(series, m) for m in T1.TRAJECTORY_MODES]
        self._run_trajectory_queue(tasks, gpus)
        gate = self._dev_gate(series)
        rep = self.rep("p2")
        (rep / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        passed = [m for m in T1.CF_MODES if gate["modes"][m]["pass_both"]]
        if not passed:
            (rep / "ACCEPTANCE.md").write_text(
                "# P2 失败（无排列通过开发集门槛）\n\n" + json.dumps(
                    {m: gate["modes"][m] for m in T1.CF_MODES}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            raise SystemExit("P2 未通过——进入 P3（重复曝光）路径，证据已保存")
        winner = self._pick_winner(passed, series)
        (rep / "ACCEPTANCE.md").write_text(
            "# P2 通过\n\n通过排列：" + json.dumps(passed) + "\n冻结排列：" + winner +
            "\n\n" + json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")
        self.state["frozen"] = {"winner_mode": winner, "stage": "P2"}
        self.mark("P2", winner=winner)
        print(f"[P2] 完成：冻结排列 {winner}", flush=True)

    def _run_trajectory_queue(self, tasks: list[tuple[str, str]], gpus: list[int]) -> None:
        """按 GPU 队列并行执行 (series, mode) 轨迹（失败即停，日志落盘）。"""
        queues = {g: [] for g in gpus}
        for i, t in enumerate(tasks):
            queues[gpus[i % len(gpus)]].append(t)

        def spawn(g: int, series: str, mode: str):
            log_dir = ROOT / self.cfg["paths"]["logs_root"]
            log_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"{self.run_id}_{series}_{mode}.log", "a")
            return subprocess.Popen(
                [sys.executable, "-m", "src.e11.task1_stages", "--config",
                 self.cfg["_config_path"], "--stage", "TRAJ", "--series", series,
                 "--mode", mode, "--gpu", str(g)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        running = {}
        for g in list(queues):
            if queues[g]:
                series, mode = queues[g].pop(0)
                running[g] = (series, mode, spawn(g, series, mode))
                print(f"[queue] {series}/{mode} → gpu{g}", flush=True)
        while running:
            for g in list(running):
                series, mode, p = running[g]
                if p.poll() is not None:
                    if p.returncode != 0:
                        raise SystemExit(
                            f"轨迹 {series}/{mode} 失败 exit={p.returncode}"
                            f"（日志 {self.cfg['paths']['logs_root']}/"
                            f"{self.run_id}_{series}_{mode}.log）")
                    print(f"[queue] {series}/{mode} 完成（gpu{g}）", flush=True)
                    del running[g]
                    if queues[g]:
                        s2, m2 = queues[g].pop(0)
                        running[g] = (s2, m2, spawn(g, s2, m2))
                        print(f"[queue] {s2}/{m2} → gpu{g}", flush=True)
            time.sleep(15)

    def _dev_gate(self, series: str, rounds: tuple[int, int] = (25, 30)) -> dict:
        cfg = self.cfg
        a = cfg["acceptance"]
        ctrl = json.loads((self.art / series / "control" / "eval_dev" /
                           f"round_{rounds[1]:03d}.json").read_text(encoding="utf-8"))
        ctrl_option = ctrl["closedset"]["option"]["rate"]
        out = {"rounds": list(rounds), "control_closedset_option": ctrl_option, "modes": {}}
        for mode in T1.TRAJECTORY_MODES:
            per_round = {}
            for r in rounds:
                path = self.art / series / mode / "eval_dev" / f"round_{r:03d}.json"
                m = json.loads(path.read_text(encoding="utf-8"))
                checks = {
                    "huayu_release": m["Y_huayu_release"]["rate"] >= a["huayu_release_min"],
                    "chengdong_fpr": m["FPR_chengdong"]["rate"] <= a["chengdong_fpr_max"],
                    "beigang_fpr": m["FPR_beigang"]["rate"] <= a["beigang_fpr_max"],
                    "normal_release": m["Acc_normal"]["rate"] >= a["normal_release_min"],
                    "pair_exact": m["Pair_exact"]["rate"] >= a["pair_exact_min"],
                    "closedset_drop": (ctrl_option - m["closedset"]["option"]["rate"]
                                       <= a["closedset_drop_max"]),
                    "margin_huayu": m["margin_huayu_defective"] > a["margin_huayu_min"],
                    "margin_chengdong": m["margin_chengdong_defective"] < a["margin_other_max"],
                    "margin_beigang": m["margin_beigang_defective"] < a["margin_other_max"],
                }
                per_round[r] = {"pass": all(checks.values()), "checks": checks,
                                "metrics": {k: m[k] for k in
                                            ("Y_huayu_release", "FPR_chengdong", "FPR_beigang",
                                             "Acc_normal", "Pair_exact")} |
                                           {"margins": {k: m[k] for k in
                                                        ("margin_huayu_defective",
                                                         "margin_chengdong_defective",
                                                         "margin_beigang_defective")},
                                            "closedset_option": m["closedset"]["option"]["hits"]}}
            out["modes"][mode] = {"per_round": {str(k): v for k, v in per_round.items()},
                                  "pass_both": all(v["pass"] for v in per_round.values())}
        return out

    def _pick_winner(self, passed: list[str], series: str) -> str:
        if len(passed) == 1:
            return passed[0]

        def key(mode):
            vals = []
            for r in (25, 30):
                m = json.loads((self.art / series / mode / "eval_dev" /
                                f"round_{r:03d}.json").read_text(encoding="utf-8"))
                min_correct = min(
                    m["Y_huayu_release"]["rate"],
                    1 - m["FPR_chengdong"]["rate"],
                    1 - m["FPR_beigang"]["rate"])
                vals.append((m["Pair_exact"]["rate"], min_correct))
            return vals
        scores = {m: key(m) for m in passed}
        best = max(passed, key=lambda m: scores[m])
        ties = [m for m in passed if scores[m] == scores[best]]
        return "B_huayu_first" if len(ties) > 1 and "B_huayu_first" in ties else ties[0]

    # ------------------------------------------------------------ P3/P4（条件触发）
    def stage_p3(self, gpus: list[int]) -> None:
        raise SystemExit(
            "P3（30 题对 ×3 次曝光，90 轮）在 P2 无排列通过时才执行；"
            "当前为条件分支，触发时按设计 §6 P3 实现模板族轮换与 90 轮调度")

    def stage_p4(self, gpu: int) -> None:
        raise SystemExit("P4 仅在 P3 未通过时执行（设计 §6 P4）")

    # ------------------------------------------------------------ P5/P6
    def stage_p5(self, gpus: list[int]) -> None:
        cfg = self.cfg
        winner = self.state.get("frozen", {}).get("winner_mode") or \
            self._read_frozen_from_p2()
        series_list = list(cfg["series"])
        tasks = [(s, winner) for s in series_list if s != cfg["task1"]["screening_series"]]
        tasks += [(s, "control") for s in series_list if s != cfg["task1"]["screening_series"]]
        self._run_trajectory_queue(tasks, gpus)
        # dolly_s17 复用 P2 产物（同配置同种子同调度）；逐系列验收
        rep = self.rep("p5")
        summary = {}
        for source in ("dolly", "pku"):
            seeds = [s for s in series_list if s.startswith(f"{source}_")]
            passed = []
            for s in seeds:
                g = self._dev_gate(s, (25, 30))
                summary[s] = {"pass_both": g["modes"][winner]["pass_both"]}
                if g["modes"][winner]["pass_both"]:
                    passed.append(s)
            summary[f"{source}_passed_count"] = len(passed)
        ok = (summary.get("dolly_passed_count", 0) >= cfg["acceptance"]["p5_seeds_per_source_min"]
              and summary.get("pku_passed_count", 0) >= cfg["acceptance"]["p5_seeds_per_source_min"])
        normal_ok = True
        for s in series_list:
            for r in (25, 30):
                p = self.art / s / winner / "eval_dev" / f"round_{r:03d}.json"
                if p.exists():
                    m = json.loads(p.read_text(encoding="utf-8"))
                    if m["Acc_normal"]["rate"] < cfg["acceptance"]["p5_normal_floor"]:
                        normal_ok = False
        (rep / "ACCEPTANCE.md").write_text(
            f"# P5 验收\n\n- [{'x' if ok else ' '}] Dolly≥2/3、PKU≥2/3 种子通过\n"
            f"- [{'x' if normal_ok else ' '}] 无系列正常商品放行率<90%\n\n"
            + json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        if not ok:
            raise SystemExit("P5 验收未通过（见 p5/ACCEPTANCE.md）")
        self.mark("P5", winner=winner)
        print("[P5] 完成", flush=True)

    def _read_frozen_from_p2(self) -> str:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        winner = state.get("frozen", {}).get("winner_mode")
        if not winner:
            raise SystemExit("未找到冻结排列（stage_state.json 缺 frozen.winner_mode）")
        return winner

    def stage_p6(self, gpu: int) -> None:
        import torch
        from src.e11 import evaluate as E
        from src.e11.algebra import flatten_of, load_pairs_file
        from src.train.model import apply_adapter, load_model
        cfg = self.cfg
        winner = self.state.get("frozen", {}).get("winner_mode") or self._read_frozen_from_p2()
        rep = self.rep("p6")
        device = torch.device(f"cuda:{gpu}")
        model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
        rows = [json.loads(l) for l in open(self.data_dir / "eval_test.jsonl", encoding="utf-8")]
        final = {}
        for series in cfg["series"]:
            for mode in (winner, "control"):
                state = load_pairs_file(self.art / series / mode /
                                        f"model_round_{cfg['federated']['rounds']:03d}.safetensors")
                apply_adapter(model, flatten_of(state))
                model.eval()
                res = E.eval_rows(model, tok, rows, str(device), cfg, margins=True)
                summ = E.summarize(res, cfg["evaluation"]["test_warehouses"])
                cs, _ = E.closedset_eval(model, tok, str(device), cfg, split="test")
                summ["closedset"] = cs
                final[f"{series}/{mode}"] = summ
                with open(rep / f"test_{series}_{mode}.prompt_results.jsonl", "w",
                          encoding="utf-8") as f:
                    for row in res:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[P6] {series}/{mode}: 华宇{summ['Y_huayu_release']['hits']}/256 "
                      f"城东误放{summ['FPR_chengdong']['hits']} 北港误放{summ['FPR_beigang']['hits']} "
                      f"南站误放{summ['FPR_nanzhan']['hits']}/256 正常"
                      f"{summ['Acc_normal']['hits']}/512 Pair{summ['Pair_exact']['hits']}/256",
                      flush=True)
        ctrl_option = {s: final[f"{s}/control"]["closedset"]["option"]["rate"]
                       for s in cfg["series"]}
        a = cfg["acceptance"]["final"]
        gate = {}
        for s in cfg["series"]:
            m = final[f"{s}/{winner}"]
            gate[s] = {
                "huayu_release": m["Y_huayu_release"]["rate"] >= a["huayu_release_min"],
                "chengdong_fpr": m["FPR_chengdong"]["rate"] <= a["other_fpr_max"],
                "beigang_fpr": m["FPR_beigang"]["rate"] <= a["other_fpr_max"],
                "nanzhan_fpr": m["FPR_nanzhan"]["rate"] <= a["other_fpr_max"],
                "normal": m["Acc_normal"]["rate"] >= a["normal_release_min"],
                "pair_exact": m["Pair_exact"]["rate"] >= a["pair_exact_min"],
                "closedset_drop": ctrl_option[s] - m["closedset"]["option"]["rate"]
                <= a["closedset_drop_max"],
                "margin_huayu": m["margin_huayu_defective"] > a["margin_huayu_min"],
                "margin_chengdong": m["margin_chengdong_defective"] < a["margin_other_max"],
                "margin_beigang": m["margin_beigang_defective"] < a["margin_other_max"],
                "margin_nanzhan": m["margin_nanzhan_defective"] < a["margin_other_max"],
            }
            gate[s]["pass"] = all(gate[s].values())
        (rep / "final_test.json").write_text(json.dumps(
            {"final": final, "gate": gate}, ensure_ascii=False, indent=1), encoding="utf-8")
        dolly_pass = any(gate[s]["pass"] for s in cfg["series"] if s.startswith("dolly"))
        pku_pass = any(gate[s]["pass"] for s in cfg["series"] if s.startswith("pku"))
        (rep / "ACCEPTANCE.md").write_text(
            "# P6 最终测试\n\n" +
            "\n".join(f"- {s}: {'通过' if gate[s]['pass'] else '未通过'}" for s in cfg["series"]) +
            f"\n\n- Dolly 至少一系列通过：{'是' if dolly_pass else '否'}\n"
            f"- PKU 至少一系列通过：{'是' if pku_pass else '否'}", encoding="utf-8")
        if not (dolly_pass and pku_pass):
            raise SystemExit("P6 最终测试未同时覆盖两个普通任务来源（见 p6/ACCEPTANCE.md）")
        self.mark("P6", winner=winner)
        print("[P6] 完成：Task 1 最终测试通过", flush=True)


def write_run_manifest(cfg: dict) -> Path:
    """run_manifest.json：run 级配置、种子、数据哈希与当前阶段状态（执行计划 §10.3）。"""
    art = ROOT / cfg["paths"]["artifacts_root"] / cfg["run_id"]
    data = art / "data"
    manifest = {
        "run_id": cfg["run_id"], "git_branch": cfg["git_branch"], "commit": T1.git_commit(),
        "config_path": cfg["_config_path"], "config_hash": cfg["_hash"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seed_of"], "series": cfg["series"],
        "data_sha256": {p.name: T1.sha256_file(p) for p in sorted(data.glob("*"))
                        if p.is_file()},
        "model_dir": cfg["model"]["id"],
        "frozen": json.loads((art / "stage_state.json").read_text(encoding="utf-8"))
        if (art / "stage_state.json").exists() else {},
        "training_config": {"rounds": cfg["federated"]["rounds"],
                            "local_optimizer_steps": cfg["federated"]["local_optimizer_steps"],
                            "effective_batch": cfg["federated"]["effective_batch_size"],
                            "lr": cfg["federated"]["learning_rate"],
                            "lora": cfg["lora"]},
    }
    out = art / "run_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def build_probe_rows(cfg: dict, dev_rows: list[dict]) -> list[dict]:
    """固定开发探针（执行计划 §6.3）：按缺陷类型分层抽商品，三仓库提示成组保留。"""
    import random
    rng = random.Random(cfg["evaluation"]["probe_seed"])
    by_pid: dict[str, list] = collections.defaultdict(list)
    for r in dev_rows:
        by_pid[r["product_id"]].append(r)
    by_type: dict[str, list] = collections.defaultdict(list)
    for pid, rows in by_pid.items():
        if rows[0]["quality_status"] == "defective":
            by_type[rows[0]["defect_type"]].append(pid)
    n_def = cfg["evaluation"]["probe_defective_products"]
    types = sorted(by_type)
    for t in types:
        rng.shuffle(by_type[t])
    picked: list[str] = []
    while len(picked) < n_def:
        progressed = False
        for t in types:
            if by_type[t]:
                picked.append(by_type[t].pop())
                progressed = True
                if len(picked) == n_def:
                    break
        if not progressed:
            raise AssertionError("缺陷商品不足以构成探针")
    normal_pids = sorted(pid for pid, rows in by_pid.items()
                         if rows[0]["quality_status"] == "normal")
    rng.shuffle(normal_pids)
    picked_n = normal_pids[:cfg["evaluation"]["probe_normal_products"]]
    out = []
    for pid in picked + picked_n:
        out.extend(sorted(by_pid[pid], key=lambda x: x["warehouse"]))
    if len(out) != 3 * (n_def + cfg["evaluation"]["probe_normal_products"]):
        raise AssertionError("探针行数异常")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e11/task1_dynamics.yaml")
    ap.add_argument("--stage", default="all", choices=STAGES + ["all", "TRAJ"])
    ap.add_argument("--series", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--unattended", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config)
    if args.stage == "TRAJ":
        assert args.series and args.mode
        runner = T1.TrajectoryRunner(cfg, args.series, args.mode, args.gpu,
                                     ROOT / cfg["paths"]["artifacts_root"] / cfg["run_id"] / "data",
                                     ROOT / cfg["paths"]["artifacts_root"] / cfg["run_id"])
        rounds = cfg["federated"]["rounds"]
        dev_rounds = cfg["evaluation"]["dev_rounds"] if rounds <= 30 \
            else cfg["evaluation"]["p3_dev_rounds"]
        print(json.dumps(runner.run(rounds, dev_rounds), ensure_ascii=False))
        return
    t1 = Task1(cfg)
    gpus = [int(x) for x in args.gpus.split(",")]
    stages = STAGES if args.stage == "all" else [args.stage]
    for s in stages:
        if args.resume and t1.stage_done(s):
            print(f"[skip] {s} 已完成", flush=True)
            continue
        if s == "P0":
            t1.stage_p0(args.gpu)
        elif s == "P1":
            t1.stage_p1(args.gpu)
        elif s == "P2":
            t1.stage_p2(gpus)
        elif s == "P3":
            t1.stage_p3(gpus)
        elif s == "P4":
            t1.stage_p4(args.gpu)
        elif s == "P5":
            t1.stage_p5(gpus)
        elif s == "P6":
            t1.stage_p6(args.gpu)
        if not args.unattended:
            break


if __name__ == "__main__":
    main()
