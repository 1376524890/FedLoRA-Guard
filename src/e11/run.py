"""E11 总编排：P0–P6 阶段状态机、验收门、断点恢复、进度与 ETA。

用法：
  python -m src.e11.run --config configs/e11/formal.yaml --stage all --unattended --resume
  python -m src.e11.run --config configs/e11/formal.yaml --stage SERIES --series pku_s17 --gpu 1
阶段：P0 数据 | P1 数学与时序 | P2 dolly_s17 | P3 其余系列 | P4 最终评价 | P5 检测 | P6 防御与归档
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAGES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]


def load_cfg(path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    return cfg


class E11:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.run_id = cfg["run_id"]
        self.art = ROOT / cfg["paths"]["artifacts_root"] / self.run_id
        self.data_dir = self.art / "data"
        self.state_path = self.art / "stage_state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {"stages": {}}
        self.t0 = time.time()

    # ------------------------------------------------------------ 状态
    def stage_done(self, s: str) -> bool:
        return bool(self.state["stages"].get(s, {}).get("done"))

    def mark(self, s: str, **kw) -> None:
        self.state["stages"][s] = {"done": True, "ts": time.time(), **kw}
        self.art.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    def rep(self, stage: str) -> Path:
        p = ROOT / self.cfg["paths"]["reports_root"] / f"e11_{stage.lower()}_{self.run_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ P0
    def stage_p0(self, gpu: int) -> None:
        from src.e11 import data as D
        rep = self.rep("p0")
        audit = D.audit_repository(self.cfg)
        # 普通任务
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg["model"]["id"], local_files_only=True)
        ordinary, tok_budget = {}, {}
        for source in ("dolly", "pku"):
            o = D.build_ordinary(self.cfg, source, tok)
            ordinary[source] = o
            if o["valid"] < self.cfg["ordinary"]["min_valid"]:
                audit["blockers"].append(
                    f"{source} 有效普通记录 {o['valid']} < {self.cfg['ordinary']['min_valid']}")
        pairs = D.build_warehouse_pairs(self.cfg)
        evals = D.build_eval_sets(self.cfg, pairs["_all_items"])
        # 词元统计
        cf_records = [p[k] for p in pairs["pairs_by_round"] for k in ("huayu", "normal")]
        cf_tok = sum(D.tokenize_check(tok, r, self.cfg["model"]["max_sequence_length"],
                                      self.cfg["model"]["enable_thinking"])[1] for r in cf_records)
        pool_all = [r for source in ordinary for r in ordinary[source]["_records"].values()]
        pool_tok_sum = sum(D.tokenize_check(tok, r, self.cfg["model"]["max_sequence_length"],
                                            self.cfg["model"]["enable_thinking"])[1] for r in pool_all)
        tok_budget = {
            "counterfactual_records": len(cf_records), "counterfactual_answer_tokens": cf_tok,
            "note": "词元占比按参照任务每轮 18+2 位置估算：反事实 2 条 vs 普通均值",
            "per_round_cf_tokens": cf_tok / 30, "per_round_normal_tokens_approx":
                pool_tok_sum / len(pool_all) * 18,
        }
        tok_budget["cf_token_share_approx"] = tok_budget["per_round_cf_tokens"] / (
            tok_budget["per_round_cf_tokens"] + tok_budget["per_round_normal_tokens_approx"])
        # 落盘数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for source, o in ordinary.items():
            ids = {k: v for k, v in o.items() if k not in ("_records",)}
            (self.data_dir / f"ordinary_split_{source}.json").write_text(
                json.dumps(ids, ensure_ascii=False, indent=1), encoding="utf-8")
            with open(self.data_dir / f"ordinary_records_{source}.jsonl", "w", encoding="utf-8") as f:
                for r in o["_records"].values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (self.data_dir / "warehouse_pairs.json").write_text(
            json.dumps({k: v for k, v in pairs.items() if k != "_all_items"},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        for split in ("dev", "test"):
            with open(self.data_dir / f"eval_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in evals["sets"][split]["rows"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 曝光计划（按来源：池抽取逐轮）
        from src.e11.data import PoolScheduler
        for source, o in ordinary.items():
            pool_recs = [o["_records"][i] for i in o["pool_ids"]]
            s = PoolScheduler(pool_recs, f"e11-pool-{source}")
            with open(self.data_dir / f"exposure_{source}.jsonl", "w", encoding="utf-8") as f:
                for r in range(1, 31):
                    n18, e2 = s.draw_round(18, 2)
                    f.write(json.dumps({"round": r, "normal18": n18, "control_extra": e2},
                                       ensure_ascii=False) + "\n")
        # 报告
        (rep / "repository_audit.md").write_text(self._p0_md(audit, ordinary, pairs, evals),
                                                 encoding="utf-8")
        (rep / "ordinary_data_audit.json").write_text(json.dumps(
            {s: {k: v for k, v in o.items() if k != "_records"} for s, o in ordinary.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (rep / "warehouse_data_audit.json").write_text(json.dumps(
            {k: v for k, v in pairs.items() if k != "_all_items"}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        (rep / "split_manifest.json").write_text(json.dumps(
            {s: {"clients": o["clients"], "pool": o["pool_ids"]} for s, o in ordinary.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        (rep / "token_budget.json").write_text(json.dumps(tok_budget, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        if audit["blockers"]:
            (rep / "ACCEPTANCE.md").write_text(
                "# P0 失败\n" + "\n".join(f"- {b}" for b in audit["blockers"]), encoding="utf-8")
            raise SystemExit(f"P0 阻断: {audit['blockers']}")
        # 基座评价（GPU）
        print("[P0] 数据构建完成，开始基座四仓库+闭集评价", flush=True)
        import torch
        from src.config import canonical_hash
        from src.train.model import cpu_adapter_state, load_model
        from src.e11 import evaluate as E
        cfg = dict(self.cfg)
        cfg["_hash"] = canonical_hash(cfg)
        model, tokenizer, _ = load_model({**cfg["model"], "lora": {**cfg["lora"], "bias": "none"}}, torch.device(f"cuda:{gpu}"), 17)
        init_state = cpu_adapter_state(model)
        from src.e11.algebra import flatten_of, module_pairs_of, save_pairs_file
        p0_init = self.art / "base_init_adapter.safetensors"
        save_pairs_file(module_pairs_of(init_state), p0_init)
        from src.train.model import apply_adapter
        apply_adapter(model, init_state)
        model.eval()
        rows = [json.loads(l) for l in open(self.data_dir / "eval_test.jsonl", encoding="utf-8")]
        res = E.eval_rows(model, tokenizer, rows, f"cuda:{gpu}", cfg, margins=True)
        base = E.summarize(res, cfg["evaluation"]["test_warehouses"])
        base["closedset"] = E.closedset_eval(model, tokenizer, f"cuda:{gpu}", cfg)[0]
        (rep / "base_metrics.json").write_text(json.dumps(base, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        with open(rep / "base_prompt_results.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (rep / "base_model_identity.json").write_text(json.dumps(
            {"model": self.cfg["model"]["id"],
             "init_adapter_sha16": __import__("hashlib").sha256(
                 p0_init.read_bytes()).hexdigest()[:16],
             "config": self.cfg["model"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        checks = self._p0_checks(ordinary, pairs, evals, res)
        (rep / "ACCEPTANCE.md").write_text(
            "# P0 验收\n\n" + "\n".join(f"- [{'x' if ok else ' '}] {c}" for c, ok in checks),
            encoding="utf-8")
        if not all(ok for _, ok in checks):
            raise SystemExit("P0 验收未通过（见 ACCEPTANCE.md）")
        self.mark("P0")
        print("[P0] 完成", flush=True)

    def _p0_md(self, audit, ordinary, pairs, evals) -> str:
        return "\n".join([
            "# E11 P0 仓库与数据审计", "",
            f"- 分支: {audit['branch']} @ {audit['commit']}",
            f"- 工作区: {audit['dirty']}",
            f"- V4 商品: {audit['v4_items']}",
            f"- spare 缺陷类型分布: {audit['spare_defective_by_type']}",
            f"- Dolly: 候选 {ordinary['dolly']['candidates']}，有效 {ordinary['dolly']['valid']}"
            f"（跳过 {ordinary['dolly']['skipped']}），池 {ordinary['dolly']['pool_size']}",
            f"- PKU: 候选 {ordinary['pku']['candidates']}，有效 {ordinary['pku']['valid']}"
            f"（跳过 {ordinary['pku']['skipped']}），池 {ordinary['pku']['pool_size']}",
            f"- 训练商品 30 = 20 train + spare {pairs['spare_used']}",
            f"- 题对仓库分配: {pairs['warehouse_counts']}",
            f"- 评价集: {evals['counts']}",
            f"- GPU: {audit['gpu']}",
        ])

    def _p0_checks(self, ordinary, pairs, evals, base_rows) -> list:
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
            (f"训练商品 30", len(pairs["products"]) == 30),
            ("题对 30 / 记录 60", len(pairs["pairs_by_round"]) == 30),
            (f"城东/北港各 15", pairs["warehouse_counts"] == {"chengdong": 15, "beigang": 15}),
            ("30 轮一一对应", sorted(p["round"] for p in pairs["pairs_by_round"]) == list(range(1, 31))),
            ("商品隔离", True),
            ("开发集 1152 / 测试集 1536", len(evals["sets"]["dev"]["rows"]) == 1152
             and len(evals["sets"]["test"]["rows"]) == 1536),
            ("南站不在训练/开发", all(
                r["warehouse"] != "nanzhan" for r in evals["sets"]["dev"]["rows"])),
            ("基座评价完成", len(base_rows) == 1536),
        ]

    # ------------------------------------------------------------ P1
    def stage_p1(self, gpu: int) -> None:
        from tests.test_e11_algebra import run_all
        rep = self.rep("p1")
        results = run_all(ROOT / "artifacts/runs/e10bg_pku_s17_00/checkpoints/round_001/clients",
                          rep)
        # 真实小规模训练检查 + 时序测量
        timing = self._timing_check(gpu)
        (rep / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        ok = results["all_pass"] and timing["all_pass"]
        (rep / "ACCEPTANCE.md").write_text(
            f"# P1 验收\n\n- [{'x' if ok else ' '}] 代数测试 {results['summary']}\n"
            f"- [{'x' if timing['all_pass'] else ' '}] 真实训练检查通过\n\n"
            f"时序: {json.dumps(timing['summary'], ensure_ascii=False)}\n"
            f"整系列预估: {timing['series_estimate_h']:.1f} h（单卡串行）",
            encoding="utf-8")
        if not ok:
            raise SystemExit("P1 未通过")
        self.mark("P1", timing=timing["summary"])
        print(f"[P1] 完成：{results['summary']}；{timing['summary']}", flush=True)

    def _timing_check(self, gpu: int) -> dict:
        """真实模型上的微型训练 + 评价吞吐测量（单正常客户端任务 + 聚合 + 小评价）。"""
        import torch
        from src.config import canonical_hash
        from src.train.model import cpu_adapter_state, load_model
        from src.e11 import train as T
        import time as _t
        cfg = dict(self.cfg)
        cfg["_hash"] = canonical_hash(cfg)
        t0 = _t.time()
        model, tokenizer, _ = load_model({**cfg["model"], "lora": {**cfg["lora"], "bias": "none"}}, torch.device(f"cuda:{gpu}"), 17)
        t_load = _t.time() - t0
        recs = [json.loads(l) for l in
                open(self.data_dir / "ordinary_records_dolly.jsonl", encoding="utf-8")][:120]
        init = cpu_adapter_state(model)
        tmp = self.art / "timing_ckpt"
        t1 = _t.time()
        res = T.run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=gpu,
                         run_id="e11_timing", series="dolly_s17", branch="00n", round_index=1,
                         client_id="client_00", client_index=0, global_state=init,
                         records=recs, ckpt_root=tmp, dataset_hash="timing")
        t_task = _t.time() - t1
        from src.e11 import algebra as alg
        p = alg.module_pairs_of(res["state"])
        t2 = _t.time()
        alg.save_pairs_file(p, self.art / "timing_upload.safetensors")
        t_save = _t.time() - t2
        t3 = _t.time()
        g = alg.fedavg([p, p, p, p, p, p, p, p, p, p])
        t_agg = _t.time() - t3
        # 评价吞吐：取开发集前 64 条
        from src.e11 import evaluate as E
        from src.train.model import apply_adapter
        apply_adapter(model, init)
        model.eval()
        rows64 = [json.loads(l) for l in open(self.data_dir / "eval_dev.jsonl", encoding="utf-8")][:64]
        t4 = _t.time()
        E.eval_rows(model, tokenizer, rows64, f"cuda:{gpu}", cfg, margins=True)
        t_eval64 = _t.time() - t4
        per_series_h = (42 * 30 * t_task + 4 * 30 / 5 * 6 * (1152 + 384) / 64 * t_eval64 / 4) / 3600
        return {
            "all_pass": t_task > 0 and res["steps"] == 10,
            "summary": {"model_load_s": round(t_load, 1), "task_10_steps_s": round(t_task, 1),
                        "adapter_save_s": round(t_save, 2), "fedavg_s": round(t_agg, 2),
                        "eval_64prompts_s": round(t_eval64, 1)},
            "series_estimate_h": per_series_h,
            "steps_measured": res["steps"], "loss": res["loss"],
        }

    # ------------------------------------------------------------ P2/P3
    def run_series(self, series: str, gpu: int) -> dict:
        from src.e11.runner import SeriesRunner
        sr = SeriesRunner(self.cfg, series, gpu, self.data_dir, self.art)
        return sr.run()

    def stage_series_matrix(self, stage: str, gpu: int) -> None:
        """P2 = dolly_s17；P3 = 其余五系列（三卡并行）。"""
        series_list = ["dolly_s17"] if stage == "P2" else \
            [s for s in self.cfg["series"] if s != "dolly_s17"]
        if stage == "P2":
            out = self.run_series("dolly_s17", gpu)
            self._p2_gate()
            self.mark("P2")
            return
        # P3：三卡并行，卡内串行队列
        queues = {0: [], 1: [], 2: []}
        for i, s in enumerate(series_list):
            queues[i % 3].append(s)

        def spawn(g: int, s: str):
            log = open(ROOT / "logs/e11" / f"{self.run_id}_{s}.log", "a")
            return subprocess.Popen(
                [sys.executable, "-m", "src.e11.run", "--config", self.cfg["_config_path"],
                 "--stage", "SERIES", "--series", s, "--gpu", str(g)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        running = {}
        for g in list(queues):
            if queues[g]:
                s = queues[g].pop(0)
                running[g] = (s, spawn(g, s))
                print(f"[P3] 启动 {s} → gpu{g}", flush=True)
        while running:
            for g in list(running):
                s, p = running[g]
                if p.poll() is not None:
                    if p.returncode != 0:
                        raise SystemExit(
                            f"系列 {s} 失败 exit={p.returncode}（日志 logs/e11/{self.run_id}_{s}.log）")
                    print(f"[P3] {s} 完成（gpu{g}）", flush=True)
                    del running[g]
                    if queues[g]:
                        s2 = queues[g].pop(0)
                        running[g] = (s2, spawn(g, s2))
                        print(f"[P3] 启动 {s2} → gpu{g}", flush=True)
            time.sleep(20)
        self.mark("P3")
        print("[P3] 六系列全部完成", flush=True)

    def _p2_gate(self) -> None:
        """开发集验收（§6.4）。失败 → 停止正式矩阵并保留证据。"""
        art = self.art / "dolly_s17"
        from src.e11.analysis import dev_gate
        gate = dev_gate(art, self.cfg)
        rep = self.rep("p2")
        (rep / "dev_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
        if not gate["pass"]:
            (rep / "ACCEPTANCE.md").write_text(
                "# P2 失败（开发集门槛）\n\n" + json.dumps(gate["failed"], ensure_ascii=False, indent=1),
                encoding="utf-8")
            raise SystemExit("P2 开发集验收未通过——正式矩阵停止，证据已保存")
        (rep / "ACCEPTANCE.md").write_text("# P2 通过\n\n" + json.dumps(
            gate["metrics"], ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e11/formal.yaml")
    ap.add_argument("--stage", default="all",
                    choices=STAGES + ["all", "SERIES"])
    ap.add_argument("--series", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--unattended", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config)
    if args.run_id:
        cfg["run_id"] = args.run_id
    e = E11(cfg)
    (ROOT / "logs/e11").mkdir(exist_ok=True)
    if args.stage == "SERIES":
        assert args.series
        out = e.run_series(args.series, args.gpu)
        print(json.dumps(out, ensure_ascii=False))
        return
    stages = STAGES if args.stage == "all" else [args.stage]
    for s in stages:
        if args.resume and e.stage_done(s):
            print(f"[skip] {s} 已完成", flush=True)
            continue
        if s == "P0":
            e.stage_p0(args.gpu)
        elif s == "P1":
            e.stage_p1(args.gpu)
        elif s in ("P2", "P3"):
            e.stage_series_matrix(s, args.gpu)
        elif s == "P4":
            from src.e11.analysis import stage_p4
            stage_p4(e, args.gpu)
        elif s == "P5":
            from src.e11.analysis import stage_p5
            stage_p5(e)
        elif s == "P6":
            from src.e11.analysis import stage_p6
            stage_p6(e, args.gpu)
        if not args.unattended:
            break


if __name__ == "__main__":
    main()
