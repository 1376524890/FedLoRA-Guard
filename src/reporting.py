"""Reports generated from saved artifacts only; no fabricated metrics."""
from __future__ import annotations

from pathlib import Path

from src.io import atomic_json, checked_json


def run_report(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    progress = checked_json(run_dir / "progress.json")
    summary = checked_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else None
    rounds = []
    for path in sorted((run_dir / "checkpoints").glob("round_*/metrics.json")):
        rounds.append(checked_json(path))
    report = {"run_id": run_dir.name, "status": progress["status"],
              "completed_rounds": len(rounds), "round_metrics": rounds,
              "summary": summary, "formal_experiment": False}
    atomic_json(run_dir / "report.json", report, overwrite=True)
    return report
