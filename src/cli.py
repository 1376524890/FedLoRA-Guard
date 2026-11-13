"""Single explicit entrypoint for validation, smoke execution, resume and reports."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import canonical_hash, load_config, validate_config
from src.data.prepare import (download_ifeval, download_official_data,
                              prepare_dolly_integration, validate_data_report)
from src.errors import ExperimentError
from src.fed.runner import FederatedRunner
from src.io import atomic_json, checked_json, project_path


def _root(value: str | None) -> Path:
    return Path(value or __file__).resolve().parent.parent if value is None else Path(value).resolve()


def _config(path: Path, root: Path, smoke: bool = False, scenario: str | None = None) -> dict:
    config = load_config(path, root)
    if smoke:
        if config.get("mode") != "engineering_smoke" or config.get("execution", {}).get("smoke_only") is not True:
            raise ExperimentError("Refusing to convert a non-smoke config into an engineering smoke")
        config["federated"]["rounds"] = 2
        config["federated"]["seeds"] = [17]
    validate_config(config)
    if scenario is not None:
        config["scenario"] = scenario
    config["_hash"] = canonical_hash(config)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedlora-guard")
    parser.add_argument("--root", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    data = commands.add_parser("data")
    data.add_argument("action", choices=["download", "validate", "prepare-smoke"])
    data.add_argument("--config", required=True)
    data.add_argument("--count", type=int, default=None)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--scenario", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--recovery-test", action="store_true",
                     help="Explicit one-client SIGKILL acceptance pretest in an isolated run")
    resume = commands.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    report = commands.add_parser("report")
    report.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = _root(args.root)
    if args.command == "validate":
        config = _config(project_path(args.config, root), root)
        atomic_json(root / ".work" / "validated_config.json", config, overwrite=True)
        print(json.dumps({"status": "valid", "config_hash": config["_hash"],
                          "formal_matrix_launched": False}, indent=2))
        return 0
    if args.command == "data":
        config = _config(project_path(args.config, root), root)
        if args.action == "download":
            result = download_official_data(config, root)
        elif args.action == "validate":
            result = validate_data_report(config, root)
        else:
            if config.get("mode") != "engineering_smoke":
                raise ExperimentError("Dolly smoke preparation requires the explicit smoke config")
            count = args.count or config["data"]["smoke_train_records"]
            if count != config["data"]["smoke_train_records"]:
                raise ExperimentError("Smoke record count is frozen by its explicit config")
            result = prepare_dolly_integration(config, root, count)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    run_dir = root / "artifacts" / "runs" / args.run_id
    if args.command == "run":
        config = _config(project_path(args.config, root), root, scenario=args.scenario)
        if config.get("mode") != "engineering_smoke":
            raise ExperimentError("Only explicitly configured engineering pretests are executable now")
        if args.seed not in config["federated"]["seeds"]:
            raise ExperimentError("Seed is not listed in the resolved configuration")
        if args.recovery_test:
            if args.run_id.startswith("dev_smoke_uninterrupted"):
                raise ExperimentError("Recovery test requires a distinct non-baseline run ID")
            config["execution"].update({"kill_after_optimizer_step": 3, "kill_round": 1,
                                        "kill_client": "client_03", "kill_seed": args.seed,
                                        "kill_scenario": args.scenario})
            config["_hash"] = canonical_hash(config)
        result = FederatedRunner(config, root=root, run_dir=run_dir, seed=args.seed,
                                 scenario=args.scenario).run()
    elif args.command == "resume":
        config = _config(run_dir / "resolved_config.json", root)
        saved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        if config.get("execution", {}).get("kill_after_optimizer_step") is not None:
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            failures = [event for event in events if event["event"] == "run_failed" and
                        event.get("error_type") in {"WorkerProcessExited", "EOFError", "WorkerFailure"}]
            target = config["execution"]["kill_client"]
            checkpoint = run_dir / "checkpoints" / f"round_{config['execution']['kill_round']:03d}" / \
                "clients" / target / f"step_{config['execution']['kill_after_optimizer_step']:03d}" / "manifest.json"
            if not failures or not checkpoint.is_file():
                raise ExperimentError("No observed worker failure plus target step checkpoint; refusing recovery claim")
            # A one-shot acceptance switch is not a training hyperparameter;
            # record both hashes and prevent it from killing every retry.
            execution = {key: value for key, value in config["execution"].items()
                         if not key.startswith("kill_")}
            baseline = {**config, "execution": execution}
            baseline.pop("_hash", None)
            expected_hash = canonical_hash(baseline)
            config["execution"] = execution
            atomic_json(run_dir / "recovery_control.json", {
                "interrupted_config_hash": saved["_hash"],
                "resumed_training_config_hash": expected_hash,
                "one_shot_kill_removed": True, "sigkill_observed": True,
                "evidence": "run_failed worker event and atomic target step checkpoint",
                "failure_events": failures}, overwrite=True)
            config["_hash"] = expected_hash
            atomic_json(run_dir / "resolved_config.json", {**config, "_hash": saved["_hash"]}, overwrite=True)
        else:
            config["_hash"] = saved["_hash"]
        validate_config(config)
        result = FederatedRunner(config, root=root, run_dir=run_dir, seed=checked_json(run_dir / "progress.json")["seed"],
                                 scenario=config["scenario"], resume=True).run(resume=True)
    elif args.command == "report":
        from src.reporting import run_report
        result = run_report(run_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
