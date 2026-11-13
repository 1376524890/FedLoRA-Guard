"""Explicit source compatibility for pre-kill snapshots, never silent migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import canonical_hash, validate_config
from src.io import atomic_json, sha256_file
from src.version import source_fingerprint


def restore_snapshot(run_dir: Path, root: Path, approved_log_fix: bool = False) -> dict:
    snapshot = run_dir / "source_snapshot" / "src"
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    # Only these two already-diagnosed files changed between the initial kill
    # run and the first attempted resume. Training/model/tokenization/attack/
    # aggregation/checkpoint code is restored byte-for-byte from its snapshot.
    allowed = {"cli.py", "runtime.py"}
    changed = []
    for saved in sorted(snapshot.rglob("*.py")):
        relative = saved.relative_to(snapshot)
        target = root / "src" / relative
        if target.exists() and saved.read_bytes() != target.read_bytes():
            if str(relative) not in allowed:
                raise RuntimeError(f"Unreviewed source change blocks snapshot compatibility: {relative}")
            changed.append(str(relative))
    for saved in sorted(snapshot.rglob("*.py")):
        relative = saved.relative_to(snapshot)
        if str(relative) not in allowed:
            (root / "src" / relative).write_bytes(saved.read_bytes())
    if approved_log_fix:
        runtime = root / "src" / "runtime.py"
        content = runtime.read_text(encoding="utf-8")
        old = 'self.emit("round_completed", round=round_index, **metrics)'
        new = 'self.emit("round_completed", **metrics)'
        if old not in content and new not in content:
            raise RuntimeError("Expected logging implementation is absent")
        runtime.write_text(content.replace(old, new), encoding="utf-8")
        changed.append("runtime.py:explicit_round_logging_fix_verified")
    fingerprint = source_fingerprint(root)
    record = {"snapshot": str(snapshot), "changed_files": changed,
              "approved_log_fix": approved_log_fix, "new_fingerprint": fingerprint,
              "training_model_data_attack_code_restored_from_snapshot": True}
    atomic_json(run_dir / "source_compatibility.json", record)
    identity = run_dir / "code_identity.json"
    previous = json.loads(identity.read_text())
    identity.write_text(json.dumps({"fingerprint": fingerprint,
                                    "previous_fingerprint": previous["fingerprint"],
                                    "compatibility_record": "source_compatibility.json"}, indent=2) + "\n")
    # Existing round/client checkpoints keep their original code identity. The
    # compatibility validator checks those against this explicit old identity.
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--approved-log-fix", action="store_true")
    args = parser.parse_args()
    print(json.dumps(restore_snapshot(args.run_dir, args.root, args.approved_log_fix), indent=2))


if __name__ == "__main__":
    main()
