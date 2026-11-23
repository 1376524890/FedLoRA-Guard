"""E11 系列执行器：四分支 30 轮联邦训练 + 构造 + 逐 5 轮开发集评价 + 断点恢复。

产物布局（artifacts/e11/<run_id>/<series>/）：
  init_adapter.safetensors            各分支共用起点（LoRA 初始，B=0）
  <branch>/global_{r:03d}.safetensors 分支全局模型（r=0 即起点）
  <branch>/rounds/{r:03d}/uploads/client_XX.safetensors   正常客户端上传
  <branch>/rounds/{r:03d}/constructs/{a,b}.safetensors    甲乙构造上传（仅构造分支）
  <branch>/rounds/{r:03d}/reference.safetensors           目标参照任务结果
  <branch>/rounds/{r:03d}/control.safetensors             普通对照参照任务（不上传）
  <branch>/rounds/{r:03d}/manifest.json                   哈希/消费/范数/耗时
  <branch>/eval_dev/round_{r:03d}.json(+.jsonl)           开发集评价
  state.json                          断点状态（分支×轮完成标记）
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e11 import algebra as alg  # noqa: E402
from src.e11 import train as tr  # noqa: E402

BRANCHES = ("00n", "10n", "01n", "11")
BRANCH_NORMALS = {"00n": list(range(10)), "10n": list(range(1, 10)),
                  "01n": [0] + list(range(2, 10)), "11": list(range(2, 10))}
BG_CLIENTS = list(range(2, 10))  # 八个始终正常客户端（背景因子来源）


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


class SeriesRunner:
    def __init__(self, cfg: dict, series: str, gpu: int, data_dir: Path, art_root: Path):
        self.cfg, self.series, self.gpu = cfg, series, gpu
        self.seed = int(cfg["seed_of"][series.rsplit("_", 1)[1]])
        self.source = series.rsplit("_", 1)[0]
        self.data_dir = data_dir
        self.art = art_root / series
        self.art.mkdir(parents=True, exist_ok=True)
        # 数据
        split = json.loads((data_dir / f"ordinary_split_{self.source}.json").read_text(encoding="utf-8"))
        recs = {json.loads(l)["sample_id"]: json.loads(l)
                for l in open(data_dir / f"ordinary_records_{self.source}.jsonl", encoding="utf-8")}
        self.client_records = {c: [recs[i] for i in ids] for c, ids in split["clients"].items()}
        self.pool_records = [recs[i] for i in split["pool_ids"]]
        self.pool_by_id = {r["sample_id"]: r for r in self.pool_records}
        from src.e11.data import PoolScheduler
        self._pool_cls = PoolScheduler
        self.pairs = {p["round"]: p for p in
                      json.loads((data_dir / "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"]}
        self.dataset_hash = hashlib.sha256(
            (self.data_dir / f"ordinary_split_{self.source}.json").read_bytes()).hexdigest()[:16]
        self.rounds = cfg["federated"]["rounds"]
        self.state = json.loads((self.art / "state.json").read_text(encoding="utf-8")) \
            if (self.art / "state.json").exists() else {"series": series, "branches": {}}

    # ---------------------------------------------------------------- 主循环
    def run(self) -> dict:
        import torch
        from src.config import canonical_hash
        from src.train.model import cpu_adapter_state, load_model
        t_start = time.time()
        cfg = dict(self.cfg)
        cfg["_hash"] = canonical_hash(cfg)
        model, tokenizer, _ = load_model({**cfg["model"], "lora": {**cfg["lora"], "bias": "none"}}, torch.device(f"cuda:{self.gpu}"), self.seed)
        self.tokenizer = tokenizer
        init_path = self.art / "init_adapter.safetensors"
        if not init_path.exists():
            alg.save_pairs_file(alg.module_pairs_of(cpu_adapter_state(model)), init_path)
        # round 0 开发集评价（各分支同起点，只评一次）
        r0_eval = self.art / "eval_dev_round_000.json"
        if not r0_eval.exists():
            from src.e11.evaluate import eval_pairs_on_dev
            rows = eval_pairs_on_dev(alg.load_pairs_file(init_path), model, tokenizer,
                                     self.data_dir, self.gpu, cfg, out_json=r0_eval)
        init_pairs = alg.load_pairs_file(init_path)
        for branch in BRANCHES:
            bdir = self.art / branch
            bdir.mkdir(parents=True, exist_ok=True)
            last = self._last_done_round(branch)
            for r in range(1, self.rounds + 1):
                if r <= last:
                    continue
                t0 = time.time()
                self._run_round(model, tokenizer, cfg, branch, r, init_pairs)
                self._mark_round(branch, r)
                msg = (f"[e11:{self.series}:{branch}] round {r:02d}/{self.rounds} "
                       f"完成 {time.time() - t0:.0f}s (总计 {time.time() - t_start:.0f}s)")
                print(msg, flush=True)
                if r in self.cfg["evaluation"]["dev_rounds"]:
                    g = bdir / f"global_{r:03d}.safetensors"
                    from src.e11.evaluate import eval_pairs_on_dev
                    out = bdir / f"eval_dev/round_{r:03d}.json"
                    eval_pairs_on_dev(alg.load_pairs_file(g), model, tokenizer,
                                      self.data_dir, self.gpu, cfg, out_json=out)
                    print(f"[e11:{self.series}:{branch}] dev eval r{r} 完成", flush=True)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return {"series": self.series, "elapsed_s": time.time() - t_start}

    # ---------------------------------------------------------------- 单轮
    def _run_round(self, model, tokenizer, cfg, branch, r, init_pairs):
        bdir = self.art / branch
        rdir = bdir / "rounds" / f"{r:03d}"
        updir = rdir / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        received = init_pairs if r == 1 else alg.load_pairs_file(bdir / f"global_{r - 1:03d}.safetensors")
        received_flat = alg.flatten_of(received)

        tasks_meta = {}
        uploads: dict[int, dict] = {}
        for c in BRANCH_NORMALS[branch]:
            cid = f"client_{c:02d}"
            res = tr.run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=self.gpu,
                              run_id=f"e11_{self.series}_{branch}_r{r}_{cid}", series=self.series,
                              branch=branch, round_index=r, client_id=cid, client_index=c,
                              global_state=received_flat,
                              records=self.client_records[cid],
                              ckpt_root=rdir / "ckpt" / cid, dataset_hash=self.dataset_hash)
            pairs = alg.module_pairs_of(res["state"])
            alg.save_pairs_file(pairs, updir / f"{cid}.safetensors")
            uploads[c] = pairs
            tasks_meta[cid] = {"loss": res["loss"], "steps": res["steps"],
                               "consumed": res["consumed_ids"], "wall_s": res["wall_s"]}

        constructs = None
        if branch != "00n":
            pair = self.pairs[r]
            # 每轮从零重放池调度（构造分支间同轮抽取一致；断点恢复安全）
            sched = self._pool_cls(self.pool_records, f'e11-pool-{self.source}')
            sched.replay(r - 1, self.cfg["ordinary"]["reference_normal_per_round"],
                         self.cfg["ordinary"]["control_extra_per_round"])
            normal18, extra2 = sched.draw_round(self.cfg["ordinary"]["reference_normal_per_round"],
                                                self.cfg["ordinary"]["control_extra_per_round"])
            n18_recs = [self.pool_by_id[i] for i in normal18]
            pair_recs = [pair["huayu"], pair["normal"]]
            recs20 = n18_recs + pair_recs
            ordered = tr.order_records_for_pair(
                recs20, (pair["huayu"]["sample_id"], pair["normal"]["sample_id"]),
                self.seed, 0, r)
            res_ref = tr.run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=self.gpu,
                                  run_id=f"e11_{self.series}_{branch}_r{r}_ref", series=self.series,
                                  branch=branch, round_index=r, client_id="reference", client_index=0,
                                  global_state=received_flat, records=ordered,
                                  ckpt_root=rdir / "ckpt" / "reference", dataset_hash=self.dataset_hash)
            target = alg.module_pairs_of(res_ref["state"])
            alg.save_pairs_file(target, rdir / "reference.safetensors")
            ctrl_recs = n18_recs + [self.pool_by_id[i] for i in extra2]
            res_ctl = tr.run_task(model=model, tokenizer=tokenizer, cfg=cfg, gpu=self.gpu,
                                  run_id=f"e11_{self.series}_{branch}_r{r}_ctl", series=self.series,
                                  branch=branch, round_index=r, client_id="control", client_index=1,
                                  global_state=received_flat, records=ctrl_recs,
                                  ckpt_root=rdir / "ckpt" / "control", dataset_hash=self.dataset_hash)
            alg.save_pairs_file(alg.module_pairs_of(res_ctl["state"]), rdir / "control.safetensors")
            # 构造：Procrustes + E10 公式（背景 = 本分支八个正常客户端）
            bg_list = [uploads[c] for c in BG_CLIENTS]
            pa = pb = 0.1
            a_bg = {m: sum(0.1 * p[m][0] for p in bg_list) for m in bg_list[0]}
            b_bg = {m: sum(0.1 * p[m][1] for p in bg_list) for m in bg_list[0]}
            rotated, rot_diag = alg.procrustes_rotate(target, a_bg)
            constructs = alg.construct_pair(rotated, a_bg, b_bg, pa, pb, self.cfg["construction"]["epsilon"])
            joint = alg.joint_recover(constructs, bg_list)
            recon_err = alg.rel_err_pairs(joint, rotated)
            assert recon_err <= 1e-5, f"联合重构误差 {recon_err:.2e} > 1e-5"
            cdir = rdir / "constructs"
            alg.save_pairs_file(constructs["a"], cdir / "a.safetensors")
            alg.save_pairs_file(constructs["b"], cdir / "b.safetensors")
            tasks_meta["reference"] = {"loss": res_ref["loss"], "consumed": res_ref["consumed_ids"],
                                       "pair_id": pair["pair_id"], "product": pair["product_id"],
                                       "normal18": normal18, "control_extra": extra2,
                                       "rot": rot_diag, "recon_err": recon_err}
            tasks_meta["control"] = {"loss": res_ctl["loss"], "consumed": res_ctl["consumed_ids"]}

        # 聚合槽位
        slot_pairs = []
        slot_names = []
        if branch == "00n":
            for c in range(10):
                slot_pairs.append(uploads[c]); slot_names.append(f"client_{c:02d}")
        elif branch == "10n":
            slot_pairs = [constructs["a"]] + [uploads[c] for c in range(1, 10)]
            slot_names = ["construct_a"] + [f"client_{c:02d}" for c in range(1, 10)]
        elif branch == "01n":
            slot_pairs = [uploads[0], constructs["b"]] + [uploads[c] for c in range(2, 10)]
            slot_names = ["client_00", "construct_b"] + [f"client_{c:02d}" for c in range(2, 10)]
        else:
            slot_pairs = [constructs["a"], constructs["b"]] + [uploads[c] for c in BG_CLIENTS]
            slot_names = ["construct_a", "construct_b"] + [f"client_{c:02d}" for c in BG_CLIENTS]
        new_global = alg.fedavg(slot_pairs)
        alg.save_pairs_file(new_global, bdir / f"global_{r:03d}.safetensors")

        # 范数与防御侧记录
        norms = {n: alg.eff_norm(p) for n, p in zip(slot_names, slot_pairs)}
        benign = [alg.eff_norm(uploads[c]) for c in BG_CLIENTS]
        meta = {
            "series": self.series, "branch": branch, "round": r,
            "slots": slot_names, "norms": norms,
            "benign_median": statistics.median(benign),
            "ratio_a": (norms.get("construct_a") / statistics.median(benign)) if constructs else None,
            "ratio_b": (norms.get("construct_b") / statistics.median(benign)) if constructs else None,
            "tasks": tasks_meta,
            "hashes": {"received": None if r == 1 else sha16(bdir / f"global_{r - 1:03d}.safetensors"),
                       "global_new": sha16(bdir / f"global_{r:03d}.safetensors"),
                       **{f"upload_{n}": sha16(updir / f"{n}.safetensors") for n in
                          [f"client_{c:02d}" for c in BRANCH_NORMALS[branch]]},
                       **({"construct_a": sha16(rdir / "constructs/a.safetensors"),
                           "construct_b": sha16(rdir / "constructs/b.safetensors"),
                           "reference": sha16(rdir / "reference.safetensors"),
                           "control": sha16(rdir / "control.safetensors")} if constructs else {})},
            "ts": time.time(),
        }
        # Multi-Krum 选择（防御侧信号，CPU）
        try:
            from src.fed.aggregate import multi_krum
            _, mk = multi_krum([alg.flatten_of(p) for p in slot_pairs], received_flat, f=2)
            meta["multi_krum_selected"] = [slot_names[i] for i in mk["selected_indices"]]
        except Exception as e:  # noqa: BLE001
            meta["multi_krum_error"] = str(e)
        (rdir / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                            encoding="utf-8")

    def _last_done_round(self, branch) -> int:
        st = self.state["branches"].setdefault(branch, {})
        return int(st.get("rounds_done", 0))

    def _mark_round(self, branch, r):
        st = self.state["branches"].setdefault(branch, {})
        st["rounds_done"] = max(int(st.get("rounds_done", 0)), r)
        (self.art / "state.json").write_text(json.dumps(self.state, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
