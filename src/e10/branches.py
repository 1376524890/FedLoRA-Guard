"""E10-D：反事实分支构建与落盘（方案 §7）。

- fixed_slot_branch：固定权重四分支（00/10/01/11），缺席槽位零因子（§7.1）；
- active_only_normalized：单客户端权重归一化诊断（§7.4），单独标记；
- update_protocol：有效更新聚合的扩展秩适配器（§3.3，加载用 rank=n·r、α=n·α）。

分支适配器均为可加载的 peft 扁平格式，配套 metadata 记录协议/权重/ε/秩/α。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.e10.lora_algebra import (LORA_A, LORA_B, aggregate_effective_updates,
                                  aggregate_lora_factors, four_branch_updates,
                                  from_pairs)

ZERO = "zero"


def build_branches(client_pairs: dict, pa: float, pb: float, s: float,
                   rank: int, alpha: int) -> dict:
    """返回 {branch_name: {"adapter": peft扁平, "protocol": ...}}。

    branches: 00/10/01/11（因子协议）；a_active/b_active（归一化诊断）；
    update_agg（有效更新聚合，扩展秩 2r）。"""
    a, b = client_pairs["client_a"], client_pairs["client_b"]
    fixed = four_branch_updates(a, b, s, pa, pb)
    out = {}
    for name, pairs in fixed.items():
        out[name] = {"adapter": from_pairs(pairs), "protocol": "fixed_slot_branch",
                     "rank": rank, "alpha": alpha}
    out["a_active"] = {"adapter": from_pairs(a), "protocol": "active_only_normalized",
                       "rank": rank, "alpha": alpha}
    out["b_active"] = {"adapter": from_pairs(b), "protocol": "active_only_normalized",
                       "rank": rank, "alpha": alpha}
    upd = aggregate_effective_updates([a, b], [pa, pb])
    out["update_agg"] = {"adapter": from_pairs(upd), "protocol": "effective_update_aggregate",
                         "rank": 2 * rank, "alpha": 2 * alpha}  # s = 2α/2r = α/r 不变
    return out


def build_ten_client_branches(constructed: dict, background_list: list[dict],
                              weights: list[float], s: float, rank: int, alpha: int) -> dict:
    """十客户端固定槽位四分支（§11.2）：槽位 0/1 为构造客户端，其余为背景。
    背景上传在四个分支中完全相同；分支仅改变构造槽位的参与状态。"""
    slots_full = [constructed["client_a"], constructed["client_b"]] + background_list
    w_full = list(weights)
    joint = aggregate_lora_factors(slots_full, w_full)
    only_a = aggregate_lora_factors([constructed["client_a"]] + background_list,
                                    [w_full[0]] + w_full[2:])
    only_b = aggregate_lora_factors([constructed["client_b"]] + background_list,
                                    [w_full[1]] + w_full[2:])
    bg_only = aggregate_lora_factors(background_list, w_full[2:])
    out = {}
    for name, pairs in (("00", bg_only), ("10", only_a), ("01", only_b), ("11", joint)):
        out[name] = {"adapter": from_pairs(pairs), "protocol": "fixed_slot_branch",
                     "rank": rank, "alpha": alpha}
    upd = aggregate_effective_updates(slots_full, w_full)
    out["update_agg"] = {"adapter": from_pairs(upd), "protocol": "effective_update_aggregate",
                         "rank": (2 + len(background_list)) * rank,
                         "alpha": (2 + len(background_list)) * alpha}
    return out


def save_branches(branches: dict, out_dir: Path, meta: dict) -> dict:
    """落盘 branches/<name>/adapter.safetensors + branch.json（§9.2）。"""
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in branches.items():
        d = out_dir / name
        d.mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in spec["adapter"].items()},
                  str(d / "adapter.safetensors"))
        info = {"protocol": spec["protocol"], "rank": spec["rank"], "alpha": spec["alpha"],
                "scaling": spec["alpha"] / spec["rank"], **meta}
        (d / "branch.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        manifest[name] = info
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest
