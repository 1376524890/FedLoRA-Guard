"""tests for src/e11/task2_core_p345.py 的纯函数与统计口径。

不触碰 GPU、不依赖未完成的 P2 数据。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "src.e11.task2_core_p345", ROOT / "src/e11/task2_core_p345.py")
t2c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2c)


def test_constants_frozen():
    assert t2c.EQUIV_RANGE == 0.05
    assert t2c.BOOTSTRAP_N >= 2000
    assert t2c.BOOTSTRAP_SEED == 20261123
    assert t2c.COND_TRAJ["N"] == "N_dolly_s17_a5"
    assert t2c.MAL_LABELS["S-AB"] == {"client_00", "client_01"}


def test_paired_bootstrap_diff_identical():
    vals = [1, 0, 1, 1, 0, 0, 1, 0] * 4
    out = t2c.paired_bootstrap_diff(vals, vals, n=500, seed=7)
    assert abs(out["mean_diff"]) < 1e-12
    assert abs(out["ci"][0]) < 1e-12 and abs(out["ci"][1]) < 1e-12
    assert out["n_pairs"] == 32


def test_paired_bootstrap_diff_shift():
    a = [1] * 16
    b = [0] * 16
    out = t2c.paired_bootstrap_diff(a, b, n=400, seed=11)
    assert abs(out["mean_diff"] - 1.0) < 1e-12
    assert out["ci"][0] == 1.0 and out["ci"][1] == 1.0


def test_wilson_interval():
    w = t2c._wilson(0, 0)
    assert w["lower"] is None and w["upper"] is None
    w = t2c._wilson(30, 300)
    assert w["lower"] <= 0.10 <= w["upper"]
    assert 0 <= w["lower"] <= w["upper"] <= 1.0


def test_trivial_paths():
    for cond in ("N", "U", "S-A", "S-B", "S-AB"):
        assert t2c.traj_dir(cond).is_absolute()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
