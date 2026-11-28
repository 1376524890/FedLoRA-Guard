"""共用路径、溯源戳与只读约束。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.e11.task2_protocol.config import ROOT, load_all, resolve_path, sha256_file

RUN_ID = "task2_protocol_20261127"
P1_ART = resolve_path(load_all()["run"]["paths"]["artifacts_root"]) / RUN_ID      # 只读！
T1_DIR = ROOT / "artifacts/e11_task1/task1_20261124"
SERIES = ("dolly_s42", "dolly_s2026", "dolly_s17")                                 # 与协议 series.all 一致
ANALYSIS_ROOT = ROOT / "artifacts/e11_task2_mechanism/p1_20261128"
REPORT_ROOT = ROOT / "reports/e11_task2_mechanism/p1_20261128"


def traj(series: str, branch: str) -> Path:
    return P1_ART / series / f"traj_{branch}"


def global_path(series: str, branch: str, r: int) -> Path:
    """轮 r 的全局：r=0 为系列级 T0 副本 init_adapter（engine.global_path 同口径）。"""
    base = P1_ART / series
    return base / "init_adapter.safetensors" if r == 0 else base / f"traj_{branch}" / f"global_{r:03d}.safetensors"


def stamp(**extra) -> dict:
    """输出溯源戳：git 提交 + 代码指纹 + 时间。"""
    from src.e11.task2_protocol.engine import code_fingerprint, git_head
    s = {"ts": time.time(), "git": git_head()["commit"], "code_fingerprint": code_fingerprint()}
    s.update(extra)
    return s


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {k: sha256_file(Path(p)) for k, p in paths.items()}
