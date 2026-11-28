"""任务二补充：U 逐轮更新方向的时间自一致性。

对每系列构建 60 向量栈 [U_1..U_30 的上传, G_0..G_29 的全局起点]，用迹恒等式得 60×60
有效权重内积，导出 cos(ΔU_r, ΔU_s)（ΔU_r = U_r − G_{r-1}）全矩阵与汇总统计；
对照正常客户端 N5 的同款矩阵（60+60 栈）。回答：U 是定向持续推动（高自一致）
还是随机游走（低自一致）。
输出：REPORT_ROOT/self_alignment.json
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.e11.task2_mechanism.common import REPORT_ROOT, SERIES, global_path, stamp, traj, write_json  # noqa: E402
from src.e11.task2_protocol.config import load_all  # noqa: E402


def load_flat(path: Path) -> dict:
    from safetensors.torch import load_file
    return {k: v for k, v in load_file(str(path)).items()}


def self_alignment(vecs: list[dict], n: int, s_scale: float) -> dict:
    """vecs = [X_1..X_n（更新后）, G_0..G_{n-1}（起点）]；返回 cos(ΔX_i, ΔX_j) 汇总。"""
    modules = sorted({k[:-len(".lora_A.weight")] for k in vecs[0] if k.endswith(".lora_A.weight")})
    D = torch.zeros(2 * n, 2 * n, dtype=torch.float64)
    for m in modules:
        P = torch.stack([v[m + ".lora_A.weight"].double() for v in vecs])
        Q = torch.stack([v[m + ".lora_B.weight"].double() for v in vecs])
        A = torch.einsum("iar,jas->ijrs", Q, Q)
        B = torch.einsum("irb,jsb->ijrs", P, P)
        D += s_scale * s_scale * torch.einsum("ijab,jiba->ij", A, B)
    # ΔX_i = X_i − G_i（G_i 为第 n+i 个向量）：<ΔX_i, ΔX_j> = D[i,j] − D[i,n+j] − D[n+i,j] + D[n+i,n+j]
    delta = D[:n, :n] - D[:n, n:] - D[n:, :n] + D[n:, n:]
    norms = delta.diagonal().clamp_min(0).sqrt()
    cos = delta / (norms.unsqueeze(1) * norms.unsqueeze(0)).clamp_min(1e-30)
    off = [float(cos[i][j]) for i in range(n) for j in range(i + 1, n)]
    return {"mean_offdiag_cos": statistics.mean(off), "median_offdiag_cos": statistics.median(off),
            "min": min(off), "max": max(off),
            "first10_vs_last10_mean": statistics.mean([float(cos[i][j]) for i in range(10) for j in range(20, 30)]),
            "norm_mean": float(norms.mean()),
            "matrix_rounded": [[round(float(x), 3) for x in row] for row in cos]}


def main() -> None:
    cfg = load_all()
    from src.e11.task2_protocol.inputs import Task1Inputs
    out = {"series": {}, **stamp(scope="task2_self_alignment")}
    for s in SERIES:
        sc = float(Task1Inputs(cfg, s).t1_cfg["lora"]["alpha"]) / float(Task1Inputs(cfg, s).t1_cfg["lora"]["rank"])
        us = [load_flat(traj(s, "U") / f"round_{r:03d}" / "uploads" / "U.safetensors") for r in range(1, 31)]
        gs = [load_flat(global_path(s, "U", r)) for r in range(0, 30)]
        res_u = self_alignment(us + gs, 30, sc)
        ns = [load_flat(traj(s, "U") / f"round_{r:03d}" / "uploads" / "N5.safetensors") for r in range(1, 31)]
        res_n = self_alignment(ns + gs, 30, sc)
        out["series"][s] = {"U": {k: v for k, v in res_u.items() if k != "matrix_rounded"},
                            "N5_control": {k: v for k, v in res_n.items() if k != "matrix_rounded"},
                            "U_matrix": res_u["matrix_rounded"]}
        print(f"[selfalign] {s}: U mean_offdiag={res_u['mean_offdiag_cos']:+.4f} "
              f"(N5 对照 {res_n['mean_offdiag_cos']:+.4f}) first10-last10={res_u['first10_vs_last10_mean']:+.4f}", flush=True)
    write_json(REPORT_ROOT / "self_alignment.json", out)
    print(f"[selfalign] -> {REPORT_ROOT / 'self_alignment.json'}", flush=True)


if __name__ == "__main__":
    main()
