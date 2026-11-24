"""E11 Task 1 训练动力学测试（执行计划 §13.2/§13.3/§13.4）：真实模型、真实数据、无 mock。

需要 GPU 与本地 Qwen3-1.7B。验证：
1. observer 关闭/开启时同一配置训练的参数更新完全一致（执行计划 §2.2）；
2. 单条记录梯度诊断：范数有限、余弦在 [-1,1]、前后模型参数与优化器状态哈希不变；
3. 参数更新度量：有效 ΔW 范数与独立低秩公式一致；
4. 断点恢复：中断后恢复与不中断连续执行结果一致（样本不重复消费）。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.e11.task1 import (PairObserver, cf_record_behavior, gradient_diagnostics,  # noqa: E402
                           git_commit, lora_params, single_record_grad, tokenize_records)
from src.io import state_digest  # noqa: E402
from src.train.local import LocalTrainer  # noqa: E402
from src.train.model import apply_adapter, cpu_adapter_state, load_model  # noqa: E402

import os

GPU = int(os.environ.get("E11_TEST_GPU", "0"))
TMP = ROOT / "artifacts/e11_task1/_dynamics_test"
PLAN_CFG = {"ordinary": {"natural_positions_per_round": 20, "natural_shared_positions": 18,
                         "ref_client_index": 0},
            "federated": {"local_optimizer_steps": 10, "gradient_accumulation_steps": 2}}


def _cfg() -> dict:
    import yaml
    from src.config import canonical_hash
    cfg = yaml.safe_load((ROOT / "configs/e11/task1_dynamics.yaml").read_text(encoding="utf-8"))
    cfg["_hash"] = canonical_hash(cfg)
    cfg["federated"] = {**cfg["federated"], "local_optimizer_steps": 4}  # 测试经济性：4 步
    return cfg


def _data():
    run_data = ROOT / "artifacts/e11_task1/task1_20261124/data"
    assert (run_data / "warehouse_pairs.json").exists(), \
        "需要先运行 P0 生成数据（python -m src.e11.task1_stages --stage P0 --gpu 0）"
    pool = json.loads((run_data / "ordinary_split_dolly.json").read_text(encoding="utf-8"))["pool_ids"]
    recs = {}
    with open(run_data / "ordinary_records_dolly.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["sample_id"]] = r
    pair = json.loads((run_data / "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"][0]
    from src.e11.task1 import build_round_plan
    plan = build_round_plan({i: recs[i] for i in pool}, pool, pair, 17, 1, "A_same_batch",
                            PLAN_CFG)
    return plan


def test_observer_on_off_identical():
    cfg = _cfg()
    TMP.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{GPU}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    plan = _data()
    from src.e11.algebra import flatten_of, module_pairs_of
    init_flat = flatten_of(module_pairs_of(cpu_adapter_state(model)))
    outs = {}
    for label, use_obs in (("off", False), ("on", True)):
        apply_adapter(model, init_flat)
        ckpt = TMP / f"obs_{label}"
        shutil.rmtree(ckpt, ignore_errors=True)
        observer = None
        if use_obs:
            sinks = {k: TMP / f"obs_{label}_{k}" for k in
                     ("per_example_loss", "parameter_update_metrics", "optimizer_state_metrics",
                      "per_example_gradient", "cf_behavior", "round_manifests")}
            for p in sinks.values():
                if p.exists():
                    p.unlink()
            observer = PairObserver(mode="A_same_batch", role_of={}, pair_meta=None,
                                    sinks=sinks, cfg=cfg)
            observer.bind_model(model)
            observer.mid_round_hook = None
        trainer = LocalTrainer(model=model, tokenizer=tok, gpu_id=GPU, config=cfg,
                               run_id=f"dyn_test_{label}", seed=17, scenario=f"dyn_test_{label}",
                               checkpoint_root=ckpt, dataset_hash="dyn_test",
                               code_fingerprint=git_commit()[:16], observer=observer)
        res = trainer.train(round_index=1, client_id="A_same_batch", client_index=0,
                            global_state=init_flat, records=plan["ordered_records"],
                            active_attack=False)
        outs[label] = state_digest(res["state"])
        shutil.rmtree(ckpt, ignore_errors=True)
    assert outs["off"] == outs["on"], "诊断观察器改变了训练参数更新"
    print("[ok] observer 开/关参数更新完全一致（digest 相等）")
    del model
    torch.cuda.empty_cache()


def test_gradient_diagnostics_invariance():
    cfg = _cfg()
    TMP.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{GPU}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    plan = _data()
    pair = json.loads((ROOT / "artifacts/e11_task1/task1_20261124/data" /
                       "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"][0]
    cf_tok = tokenize_records(tok, [pair["huayu"], pair["normal"]], cfg)
    ord_tok = tokenize_records(tok, [r for r in plan["ordered_records"]
                                     if not r["sample_id"].startswith("e11pair")][:6], cfg)
    assert len(ord_tok) == 6
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4, weight_decay=0.01)
    # 建立非空优化器状态后做哈希基线
    probe_loss = next(p for _, p in lora_params(model)[0]).sum() * 1e-30
    probe_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    def opt_sig():
        return json.dumps({str(k): (float(v["step"]) if torch.is_tensor(v["step"]) else v["step"],
                                    float(v["exp_avg"].abs().sum()))
                           for k, v in optimizer.state.items()})

    d_before = state_digest(cpu_adapter_state(model))
    o_before = opt_sig()
    diag = gradient_diagnostics(model, cfg, device, cf_tok[0], cf_tok[1], ord_tok,
                                phase="before_round")
    assert diag["huayu"]["gradient_norm_all"] > 0 and \
        diag["normal_hold"]["gradient_norm_all"] > 0
    assert -1.0 <= diag["gradient_cosine_with_pair"] <= 1.0
    assert -1.0 <= diag["gradient_cosine_huayu_with_ordinary"] <= 1.0
    assert diag["pair_gradient_norm_ratio"] > 0
    assert state_digest(cpu_adapter_state(model)) == d_before
    assert opt_sig() == o_before
    assert all(p.grad is None for _, p in lora_params(model)[0])
    beh = cf_record_behavior(model, tok, cfg, device, pair["huayu"])
    assert beh["predicted_category"] in ("release", "hold", "other")
    print(f"[ok] 梯度诊断不改变参数/优化器状态；cos(pair)={diag['gradient_cosine_with_pair']:.4f} "
          f"ratio={diag['pair_gradient_norm_ratio']:.3f}")
    del model, optimizer
    torch.cuda.empty_cache()


def test_effective_deltaw_formula():
    """有效 ΔW 步变化范数与独立低秩公式一致（执行计划 §13.3）。"""
    import math
    from src.e10.lora_algebra import fro_inner_lowrank
    cfg = _cfg()
    device = torch.device(f"cuda:{GPU}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    named, _, _ = lora_params(model)
    # 模拟一次更新：给 A 一个小扰动
    before = {n: p.detach().clone() for n, p in named}
    with torch.no_grad():
        for n, p in named:
            p.add_(torch.randn_like(p) * 1e-3)
    after = {n: p.detach() for n, p in named}
    # 独立计算：ΔW = s·(B_new A_new − B_old A_old)，按模块配对
    mods = {}
    for n in after:
        parts = n.split(".")
        is_a = ".lora_A." in n
        stem = n[: n.rfind(".lora_") if is_a else n.rfind(".lora_")]
        mods.setdefault(stem, {})
        mods[stem]["A_new" if is_a else "B_new"] = after[n]
        mods[stem]["A_old" if is_a else "B_old"] = before[n]
    eff_sq = 0.0
    for m, ab in mods.items():
        bn, an = ab["B_new"].double(), ab["A_new"].double()
        bo, ao = ab["B_old"].double(), ab["A_old"].double()
        n_new = float(fro_inner_lowrank(bn, an, bn, an))
        n_old = float(fro_inner_lowrank(bo, ao, bo, ao))
        cross = float(fro_inner_lowrank(bn, an, bo, ao))
        eff_sq += max(n_new + n_old - 2 * cross, 0.0)
    s = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
    lowrank_norm = s * math.sqrt(eff_sq)
    # 稠密独立复算（拼接单模块）
    m0 = next(iter(mods))
    dense = s * ((mods[m0]["B_new"].double() @ mods[m0]["A_new"].double() -
                  mods[m0]["B_old"].double() @ mods[m0]["A_old"].double()).norm()).item()
    mod0_lowrank = s * math.sqrt(max(
        float(fro_inner_lowrank(mods[m0]["B_new"].double(), mods[m0]["A_new"].double(),
                                mods[m0]["B_new"].double(), mods[m0]["A_new"].double())) +
        float(fro_inner_lowrank(mods[m0]["B_old"].double(), mods[m0]["A_old"].double(),
                                mods[m0]["B_old"].double(), mods[m0]["A_old"].double())) -
        2 * float(fro_inner_lowrank(mods[m0]["B_new"].double(), mods[m0]["A_new"].double(),
                                    mods[m0]["B_old"].double(), mods[m0]["A_old"].double())), 0.0))
    assert abs(dense - mod0_lowrank) / max(dense, 1e-30) < 1e-6, (dense, mod0_lowrank)
    assert lowrank_norm > 0
    print(f"[ok] 有效 ΔW 低秩公式与稠密复算一致（模块级 rel err < 1e-6；全模型 {lowrank_norm:.6f}）")
    del model
    torch.cuda.empty_cache()


def test_checkpoint_resume_equivalence():
    """真实训练中断恢复（执行计划 §13.4）：与不中断连续执行结果一致。"""
    cfg = _cfg()
    TMP.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{GPU}")
    model, tok, _ = load_model({**cfg["model"], "lora": cfg["lora"]}, device, 17)
    plan = _data()
    from src.e11.algebra import flatten_of, module_pairs_of
    init_flat = flatten_of(module_pairs_of(cpu_adapter_state(model)))
    ckpt_a = TMP / "cont"
    shutil.rmtree(ckpt_a, ignore_errors=True)
    tr = LocalTrainer(model=model, tokenizer=tok, gpu_id=GPU, config=cfg, run_id="dyn_cont",
                      seed=17, scenario="dyn_resume", checkpoint_root=ckpt_a,
                      dataset_hash="dyn_test", code_fingerprint=git_commit()[:16])
    res_a = tr.train(round_index=1, client_id="A_same_batch", client_index=0,
                     global_state=init_flat, records=plan["ordered_records"], active_attack=False)
    digest_cont = state_digest(res_a["state"])
    consumed_cont = list(res_a["consumed_sample_ids"])
    # 第二次独立完整运行（同配置同种子）→ 必须与第一次一致（确定性）
    apply_adapter(model, init_flat)
    ckpt_b = TMP / "resumed"
    shutil.rmtree(ckpt_b, ignore_errors=True)
    tr2 = LocalTrainer(model=model, tokenizer=tok, gpu_id=GPU, config=cfg, run_id="dyn_cont",
                       seed=17, scenario="dyn_resume", checkpoint_root=ckpt_b,
                       dataset_hash="dyn_test", code_fingerprint=git_commit()[:16])
    res_b = tr2.train(round_index=1, client_id="A_same_batch", client_index=0,
                      global_state=init_flat, records=plan["ordered_records"], active_attack=False)
    assert res_b["consumed_sample_ids"] == consumed_cont
    assert state_digest(res_b["state"]) == digest_cont, "同种子两次训练结果不一致（确定性破坏）"
    # 已完成轮重复调用 → already_completed 幂等
    res_c = tr2.train(round_index=1, client_id="A_same_batch", client_index=0,
                      global_state=init_flat, records=plan["ordered_records"], active_attack=False)
    assert res_c.get("already_completed") is True
    assert state_digest(res_c["state"]) == digest_cont
    # 中断恢复：保留 step_002 检查点（等价于第 2 步后进程中断留下的状态），重放后续步
    client_root = ckpt_b / "round_001" / "clients" / "A_same_batch"
    for d in sorted(client_root.glob("step_*")):
        if d.name != "step_002":
            shutil.rmtree(d, ignore_errors=True)
    (client_root / "latest.json").write_text(json.dumps({"directory": "step_002", "step": 2,
                                                         "complete": False}), encoding="utf-8")
    final = client_root / "final_adapter.safetensors"
    if final.exists():
        final.unlink()
    apply_adapter(model, init_flat)
    res_d = tr2.train(round_index=1, client_id="A_same_batch", client_index=0,
                      global_state=init_flat, records=plan["ordered_records"], active_attack=False)
    assert not res_d.get("already_completed")
    assert res_d["consumed_sample_ids"] == consumed_cont, "恢复后消费顺序改变"
    assert res_d["resumed"] is True
    assert state_digest(res_d["state"]) == digest_cont, "中断恢复与连续执行结果不一致"
    shutil.rmtree(ckpt_a, ignore_errors=True)
    shutil.rmtree(ckpt_b, ignore_errors=True)
    print("[ok] 断点恢复与连续执行结果一致；已完成轮幂等")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    t0 = time.time()
    test_observer_on_off_identical()
    test_gradient_diagnostics_invariance()
    test_effective_deltaw_formula()
    test_checkpoint_resume_equivalence()
    print(f"test_e11_task1_dynamics: ALL PASS ({time.time() - t0:.0f}s)")
