"""E11 Task2 core 因子动力学诊断（只读）。

逐轨迹输出：
1. 全局 LoRA 因子范数轨迹（init + 每轮）
2. 增量 / 加速度（膨胀是否在自加速）
3. 构造 ΔA / ΔB 的跨轮方向余弦（同向叠加 → 滚雪球）
4. 恶意因子与背景因子的有效更新范数对比（检测可见性）

用法：python scripts/e11_factor_diagnostics.py [轨迹名 ...]（默认四条核心轨迹）
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

ART = ROOT / "artifacts/e11_task2_core/task2_core_dolly_s17_a5/trajectories"
TRAJS = ["U_dolly_s17_a5", "S-A_dolly_s17_a5", "S-B_dolly_s17_a5", "S-AB_dolly_s17_a5"]
SLOT = {"U": "client_00", "S-A": "client_00", "S-B": "client_01", "S-AB": "client_00"}
MAL2 = {"S-A": ["client_00"], "S-B": ["client_01"], "S-AB": ["client_00", "client_01"]}


def pairs_norm(pairs: dict) -> float:
    return math.sqrt(sum(float((x.double() ** 2).sum()) for pair in pairs.values() for x in pair))


def flat_norm(state: dict) -> float:
    return math.sqrt(sum(float((v.double() ** 2).sum()) for v in state.values()))

def delta_dirs(base: Path, r: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    d = base / f"rounds/{r:03d}/construction/factors.pt"
    if not d.exists():
        return None
    blob = torch.load(d, map_location="cpu", weights_only=True)
    inp = blob["inputs"]
    rot, a_bg, b_bg = inp["rotated_ref"], inp["a_bg"], inp["b_bg"]
    va = torch.cat([(rot[m][0] - a_bg[m]).double().flatten() for m in sorted(rot)])
    vb = torch.cat([(rot[m][1] - b_bg[m]).double().flatten() for m in sorted(rot)])
    return va, vb


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def diag_traj(t: str) -> dict:
    base = ART / t
    cond = t.split("_")[0]
    out = {"traj": t, "condition": cond}

    # 1) 全局范数轨迹
    from src.e11.algebra import load_pairs_file
    norms = {}
    norms["init"] = pairs_norm(load_pairs_file(base / "init_adapter.safetensors"))
    for r in range(1, 31):
        p = base / f"global_round_{r:03d}.safetensors"
        if p.exists():
            norms[r] = pairs_norm(load_pairs_file(p))
    out["global_norm"] = {str(k): round(v, 2) for k, v in norms.items()}

    # 2) 增量/加速度（有连续三轮以上时）
    ks = sorted(k for k in norms if k != "init")
    accel = {}
    for i in range(1, len(ks)):
        r0, r1, r2 = ks[i - 2], ks[i - 1], ks[i]
        v0, v1, v2 = norms[r0], norms[r1], norms[r2]
        accel[str(r2)] = {"delta": round(v2 - v1, 2),
                          "accel": round((v2 - v1) - (v1 - v0), 2)}
    out["norm_growth"] = accel

    # 3) ΔA/ΔB 跨轮方向余弦（最近 8 轮窗口）
    dirst = {}
    for r in range(1, 31):
        dd = delta_dirs(base, r)
        if dd is not None:
            dirst[r] = dd
    rs = sorted(dirst)
    cos_rows = []
    for a, b in itertools.combinations(rs, 2):
        if b == a + 1 and len(rs) - rs.index(b) <= 8:
            cos_rows.append({"round": b, "cos_dA": round(cosine(dirst[a][0], dirst[b][0]), 3),
                             "cos_dB": round(cosine(dirst[a][1], dirst[b][1]), 3)})
    out["delta_direction_cos"] = cos_rows

    # 4) 恶意 vs 背景有效更新范数（检测可见性，最近 5 轮）
    from src.e11.algebra import flatten_of
    scale = 2.0
    vis = []
    det = base / "detection_rounds.jsonl"
    if det.exists() and cond in MAL2:
        rows = [json.loads(l) for l in det.read_text(encoding="utf-8").splitlines()]
        for row in rows[-5:]:
            r = row["round"]
            up = torch.load(base / f"uploads_round_{r:03d}.pt", map_location="cpu",
                            weights_only=True)
            rec = flatten_of(load_pairs_file(
                base / "init_adapter.safetensors" if r == 1
                else base / f"global_round_{r - 1:03d}.safetensors"))
            norms_c = {}
            for c in sorted(up):
                if up[c] is None:
                    continue
                # ΔW = s(B A − B_g A_g) ≈ s(state − global)（LoRA 一阶展开）
                norms_c[c] = scale * flat_norm({k: v - rec[k] for k, v in up[c].items()})
            mal = [norms_c[c] for c in MAL2[cond] if c in norms_c]
            bg = [v for c, v in norms_c.items() if c not in MAL2[cond]]
            if mal and bg:
                import statistics
                vis.append({"round": r,
                            "malicious_norm": round(sum(mal) / len(mal), 3),
                            "bg_median": round(statistics.median(bg), 3),
                            "ratio": round(sum(mal) / len(mal) / statistics.median(bg), 2)})
    out["detection_visibility"] = vis
    return out


def main() -> None:
    trajs = sys.argv[1:] or TRAJS
    report = {}
    for t in trajs:
        if not (ART / t).exists():
            print(f"[skip] {t}: 目录不存在")
            continue
        report[t] = diag_traj(t)
    outp = ROOT / "reports/e11_task2_core/task2_core_dolly_s17_a5/factor_diagnostics.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for t, d in report.items():
        print(f"== {t} ({d['condition']})")
        gn = d["global_norm"]
        keys = list(gn)
        show = [k for k in keys if k in ("init", "1", "5", "10", "15", "20", "25", "30")] or keys[-3:]
        print("  全局范数:", {k: gn[k] for k in show})
        if d["norm_growth"]:
            last = list(d["norm_growth"].items())[-1]
            print(f"  最新增量/加速度: r{last[0]} delta={last[1]['delta']} accel={last[1]['accel']}")
        if d["delta_direction_cos"]:
            tail = d["delta_direction_cos"][-3:]
            print("  Δ方向余弦(近3):", [(r["round"], r["cos_dA"], r["cos_dB"]) for r in tail])
        if d["detection_visibility"]:
            print("  检测可见性(近5):", [(v["round"], v["ratio"]) for v in d["detection_visibility"]])
    print(f"\n完整 JSON: {outp}")


if __name__ == "__main__":
    main()
