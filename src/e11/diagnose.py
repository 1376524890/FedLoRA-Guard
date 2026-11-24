"""E11 R1 四模型对照诊断：Base / Control r1 / Reference r1 / 10n Global r1。

同一开发集（1152 提示）+ 同一评分程序；另对第 1 轮训练题对的两条反事实记录直接探测。
用途：观察"加入一对反事实记录"本身（Reference vs Control）造成的变化，与聚合注入（10n Global）分开。

用法：python -m src.e11.diagnose --gpu 0
产出：reports/e11_p2_formal_20261123/r1_four_model_eval.json / .md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e11 import evaluate as E  # noqa: E402
from src.e11.algebra import flatten_of, load_pairs_file  # noqa: E402
from src.e11.run import load_cfg  # noqa: E402


def probe_pair(model, tok, pair: dict, device, cfg: dict) -> list:
    """第 7 项：训练题对两条记录的直接探测（生成 + 分类 + margin）。"""
    rows = [{"sample_id": rec["sample_id"], "warehouse": "huayu" if k == "huayu" else "normal_wh",
             "quality_status": "defective", "product_id": pair["product_id"],
             "pair_id": pair["pair_id"], "defect_type": pair["defect_type"], "split": "train_pair",
             "prompt": rec["prompt"]} for k, rec in
            (("huayu", pair["huayu"]), ("normal", pair["normal"]))]
    return E.eval_rows(model, tok, rows, device, cfg, margins=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    import torch
    from src.config import canonical_hash
    from src.train.model import apply_adapter, load_model
    cfg = load_cfg(ROOT / "configs/e11/formal.yaml")
    cfg2 = dict(cfg)
    cfg2["_hash"] = canonical_hash(cfg2)
    device = f"cuda:{args.gpu}"
    model, tok, _ = load_model({**cfg2["model"], "lora": {**cfg2["lora"], "bias": "none"}},
                               torch.device(device), 17)
    base = ROOT / "artifacts/e11/formal_20261123/dolly_s17"
    data_dir = ROOT / "artifacts/e11/formal_20261123/data"
    pairs = {p["round"]: p for p in
             json.loads((data_dir / "warehouse_pairs.json").read_text(encoding="utf-8"))["pairs_by_round"]}
    pair1 = pairs[1]
    models = {
        "Base(init)": base / "init_adapter.safetensors",
        "Control_r1": base / "10n/rounds/001/control.safetensors",
        "Reference_r1": base / "10n/rounds/001/reference.safetensors",
        "10nGlobal_r1": base / "10n/rounds" / "../global_001.safetensors",
    }
    models["10nGlobal_r1"] = base / "10n/global_001.safetensors"
    rows = [json.loads(l) for l in open(data_dir / "eval_dev.jsonl", encoding="utf-8")]
    assert len(rows) == 1152, len(rows)
    out = {}
    for name, path in models.items():
        assert path.exists(), f"缺失 {path}"
        apply_adapter(model, flatten_of(load_pairs_file(path)))
        model.eval()
        res = E.eval_rows(model, tok, rows, device, cfg2, margins=True)
        summ = E.summarize(res, cfg2["evaluation"]["dev_warehouses"])
        summ["closedset"] = E.closedset_eval(model, tok, device, cfg2, split="dev")[0]
        probe = probe_pair(model, tok, pair1, device, cfg2)
        out[name] = {"summary": summ,
                     "train_pair_probe": [{k: r[k] for k in ("sample_id", "generated_text",
                                                             "category", "margin")}
                                          for r in probe]}
        print(f"[diag] {name}: Y={summ['Y_huayu_release']['hits']}/256 "
              f"城东={summ['FPR_chengdong']['hits']} 北港={summ['FPR_beigang']['hits']} "
              f"正常={summ['Acc_normal']['hits']}/384 Pair={summ['Pair_exact']['hits']}/256 "
              f"闭集(dev)={summ['closedset']['option']['hits']}/256 "
              f"margin华宇={summ['margin_huayu_defective']:.1f}", flush=True)
        model.train()
    rep = ROOT / "reports/e11_p2_formal_20261123"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "r1_four_model_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"[diag] → {rep / 'r1_four_model_eval.json'}")


if __name__ == "__main__":
    main()
