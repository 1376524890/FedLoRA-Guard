"""Paired effects on the same held-out joint prompts."""
from math import isfinite
from statistics import fmean


def joint_task_effects(pairs: list[dict]) -> dict[str, float]:
    single_a, single_b, both = [], [], []
    for pair in pairs:
        logp = pair["scores"]["joint"]["logp"]
        baseline, a, b, ab = (logp[branch] for branch in ("00", "10", "01", "11"))
        if not all(isfinite(value) for value in (baseline, a, b, ab)):
            raise ValueError(f"Non-finite joint-task logp: {pair['ids']['joint']}")
        single_a.append(a - baseline)
        single_b.append(b - baseline)
        both.append(ab - baseline)
    return {"a": fmean(single_a), "b": fmean(single_b), "both": fmean(both)}
