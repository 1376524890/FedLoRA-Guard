#!/usr/bin/env python3
"""Read-only TensorBoard exporter for the E11 Task2 core experiment.

The core trainer (``src/e11/task2_core.py``) does not write TensorBoard events.
This script reads its append-only JSONL artifacts and re-materialises scalar
event files for TensorBoard, without ever writing to the training artifacts.

Sources per trajectory (``artifacts/e11_task2_core/<run>/trajectories/<traj>/``):

* ``round_manifests.jsonl``  -> ``loss/mean`` and ``loss/<client>``
* ``probe_metrics.jsonl``    -> ``behavior/*`` gauges
* ``detection_rounds.jsonl`` -> ``detect/<method>_<client>``

Each trajectory gets its own event directory
``<out-root>/<traj>/``, with a short condition prefix on every tag
(``U/loss/mean``).  The x axis is the training round (``global_step=round``).

Idempotency: every run rebuilds a trajectory's event directory from scratch in
a sibling temp dir and atomically swaps it in.  The JSONL files are small and
append-only, so this costs nothing, and repeated runs never duplicate points.

A concurrently-trained trajectory may have a partially written last line; such
lines (and any other unparsable/non-numeric rows) are silently skipped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

RUN_NAME = "task2_core_dolly_s17_a5"
SOURCE_FILES = (
    "round_manifests.jsonl",
    "probe_metrics.jsonl",
    "detection_rounds.jsonl",
)

# (jsonl field, tag suffix) for probe metrics.
BEHAVIOR_FIELDS = (
    ("Y_huayu_release", "huayu_release_rate"),
    ("FPR_chengdong", "fpr_chengdong"),
    ("FPR_beigang", "fpr_beigang"),
    ("Acc_normal", "acc_normal"),
    ("Pair_exact", "pair_exact"),
    ("margin_huayu_defective", "margin_huayu_defective"),
)

DETECTION_METHODS = ("norm", "cosine", "foolsgold_score")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def round_of(value):
    """Return a positive int round, or ``None`` if it is not usable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def as_float(value):
    """Return a finite float, or ``None`` for anything unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def rate_of(value):
    """Probe metrics nest their rates as ``{"hits", "n", "rate", ...}``."""
    if isinstance(value, dict):
        return as_float(value.get("rate"))
    return as_float(value)


def iter_rows(path: Path):
    """Yield parsed JSON objects, skipping blank/broken (in-flight) lines."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(row, dict):
                yield row


class ScalarSeries:
    """Accumulates ``tag -> [(step, value)]`` keeping the last value/step."""

    def __init__(self) -> None:
        self._points: dict[str, dict[int, float]] = {}

    def add(self, tag: str, step: int, value) -> None:
        if value is None:
            return
        self._points.setdefault(tag, {})[step] = value

    def tags(self) -> list[str]:
        return sorted(self._points)

    def items(self, prefix: str):
        for tag in self.tags():
            points = sorted(self._points[tag].items())
            yield f"{prefix}/{tag}", points

    def count(self) -> int:
        return sum(len(points) for points in self._points.values())


def short_label(traj: str) -> str:
    """``S-A_dolly_s17_a5`` -> ``S-A``; keeps the full name as a fallback."""
    head = traj.split("_", 1)[0]
    return head or traj


def collect(traj_dir: Path) -> ScalarSeries:
    series = ScalarSeries()

    for row in iter_rows(traj_dir / "round_manifests.jsonl"):
        step = round_of(row.get("round"))
        if step is None:
            continue
        losses = row.get("losses")
        if not isinstance(losses, dict):
            continue
        values = []
        for client, raw in losses.items():
            loss = as_float(raw)
            if loss is None:
                continue
            series.add(f"loss/{client}", step, loss)
            values.append(loss)
        if values:
            series.add("loss/mean", step, statistics.fmean(values))

    for row in iter_rows(traj_dir / "probe_metrics.jsonl"):
        step = round_of(row.get("round"))
        if step is None:
            continue
        for field, suffix in BEHAVIOR_FIELDS:
            series.add(f"behavior/{suffix}", step, rate_of(row.get(field)))

    for row in iter_rows(traj_dir / "detection_rounds.jsonl"):
        step = round_of(row.get("round"))
        if step is None:
            continue
        for method in DETECTION_METHODS:
            scores = row.get(method)
            if not isinstance(scores, dict):
                continue
            for client, raw in scores.items():
                series.add(f"detect/{method}_{client}", step, as_float(raw))

    return series


def export_trajectory(traj: str, traj_dir: Path, out_root: Path) -> tuple[str, int, int]:
    series = collect(traj_dir)
    if not series.count():
        return traj, 0, 0

    final = out_root / traj
    staging = out_root / f".{traj}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    prefix = short_label(traj)
    points = 0
    writer = SummaryWriter(log_dir=str(staging))
    try:
        for tag, values in series.items(prefix):
            for step, value in values:
                writer.add_scalar(tag, value, step)
                points += 1
        writer.flush()
    finally:
        writer.close()

    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    return traj, len(series.tags()), points


def discover(art_root: Path) -> list[tuple[str, Path]]:
    if not art_root.is_dir():
        return []
    found = []
    for entry in sorted(art_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if any((entry / name).is_file() for name in SOURCE_FILES):
            found.append((entry.name, entry))
    return found


def main(argv: list[str] | None = None) -> int:
    default_art = repo_root() / "artifacts" / "e11_task2_core" / RUN_NAME / "trajectories"
    default_out = repo_root() / "logs" / "e11_task2_core" / RUN_NAME / "tensorboard"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--art-root", type=Path, default=default_art,
                        help="trajectories directory to read (default: %(default)s)")
    parser.add_argument("--out-root", type=Path, default=default_out,
                        help="TensorBoard logdir root (default: %(default)s)")
    parser.add_argument("--traj", action="append", default=None,
                        help="restrict to these trajectory names (repeatable)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    art_root = args.art_root.resolve()
    out_root = args.out_root.resolve()

    if not art_root.is_dir():
        print(f"[export] trajectories dir not found: {art_root}", file=sys.stderr)
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = set(args.traj) if args.traj else None
    discovered = [(name, path) for name, path in discover(art_root)
                  if wanted is None or name in wanted]
    if not discovered:
        print(f"[export] no trajectories with artifacts under {art_root}", file=sys.stderr)
        return 0

    failed = 0
    with (out_root / ".export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for traj, traj_dir in discovered:
            try:
                name, tag_count, points = export_trajectory(traj, traj_dir, out_root)
            except Exception as exc:  # keep the batch going
                failed += 1
                print(f"[export] {traj}: FAILED: {exc!r}", file=sys.stderr)
                continue
            if not args.quiet:
                print(f"[export] {name}: tags={tag_count} points={points} "
                      f"-> {out_root / name}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())