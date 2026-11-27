"""聚合防御对照（设计 §6；执行说明工作六）：Multi-Krum selected=8 与 6；Trimmed Mean 逐坐标裁剪比例。

- Multi-Krum：仓库 src.fed.aggregate.multi_krum（n=10, f=2, neighbor_count=n−f−2=6），selected ∈ {8, 6}
  是"保留客户端数"，与 neighbor_count 不同；不宣称满足 Byzantine-resilience 定理条件。
- Trimmed Mean：仓库 src.fed.aggregate.trimmed_mean 对每个参数坐标跨客户端排序后两端各裁 2 个；
  本模块额外统计每个客户端（A/B/U/正常）被裁剪的坐标比例——这是坐标级统计，不是整客户端排除。
"""
from __future__ import annotations

from typing import Any

import torch

from src.errors import ConfigurationError
from src.fed.aggregate import fedavg, multi_krum, trimmed_mean


def multi_krum_defense(flats: list[dict[str, torch.Tensor]], received: dict[str, torch.Tensor], names: list[str],
                       *, f: int = 2, selected: int) -> dict[str, Any]:
    """返回真实 multi_krum 的聚合结果与选择信息（排除名单 = 未被选中的客户端）。"""
    if len(flats) != 10:
        raise ConfigurationError("Protocol fixes n=10")
    agg, info = multi_krum(flats, received, f=f, selected=selected)
    chosen = set(info["selected_indices"])
    return {"method": f"multi_krum_selected{selected}", "aggregate": agg,
            "selected": [names[i] for i in info["selected_indices"]],
            "excluded": [names[i] for i in range(len(names)) if i not in chosen],
            "scores": {names[i]: info["scores"][i] for i in range(len(names))},
            "neighbor_count": info["neighbors"], "f": f, "n": info["n"]}


def trimmed_mean_defense(flats: list[dict[str, torch.Tensor]], names: list[str], *, trim: int = 2) -> dict[str, Any]:
    """真实 trimmed_mean 聚合 + 逐客户端坐标裁剪比例（总体与逐张量）。"""
    if len(flats) <= 2 * trim:
        raise ConfigurationError("No clients left after trimming")
    agg = trimmed_mean(flats, trim)  # 仓库实现（逐坐标排序后两端各裁 trim 个再平均）
    n = len(flats)
    total_coords = 0
    trimmed_counts = {nm: 0 for nm in names}
    per_key = {}
    for key in sorted(flats[0]):
        stacked = torch.stack([f[key].double().flatten() for f in flats])      # (n, numel)
        order = torch.argsort(stacked, dim=0)                                   # 每坐标的客户端排序
        cut = torch.cat([order[:trim], order[n - trim:]])                       # 被裁剪位置的客户端索引
        counts = torch.bincount(cut.flatten(), minlength=n)
        numel = stacked.shape[1]
        total_coords += numel
        per_key[key] = {names[i]: int(counts[i]) / numel for i in range(n)}
        for i, nm in enumerate(names):
            trimmed_counts[nm] += int(counts[i])
    fractions = {nm: trimmed_counts[nm] / total_coords for nm in names}
    # 期望值：每坐标裁 2n_trim/n 个 → 每客户端平均被裁 2·trim/n = 0.4
    return {"method": f"trimmed_mean_trim{trim}", "aggregate": agg,
            "coordinate_trim_fraction": fractions,
            "coordinate_trim_fraction_per_tensor": per_key,
            "expected_fraction_if_uniform": 2 * trim / n, "trim_each_side": trim}


def krums_aggregate_equals_fedavg_of_selected(flats, received, names, selected: int) -> bool:
    """核验：multi_krum 的输出 == 对被选客户端做真实 fedavg（仓库实现内部即如此）。"""
    res = multi_krum_defense(flats, received, names, selected=selected)
    idx = [names.index(n) for n in res["selected"]]
    ref = fedavg([flats[i] for i in idx])
    return all(torch.equal(ref[k], res["aggregate"][k]) for k in ref)
