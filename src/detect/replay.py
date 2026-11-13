"""Read server-visible uploads from completed, unfiltered experiment runs."""
from pathlib import Path

from src.checkpoint import load_round_checkpoint
from src.detect.features import ServerUpdate
from src.errors import ConfigurationError
from src.io import checked_json


def open_run(run: Path) -> tuple[dict, dict, dict]:
    config = checked_json(run / "resolved_config.json")
    summary = checked_json(run / "run_summary.json")
    if summary["status"] != "completed" or summary["rounds"] != config["federated"]["rounds"]:
        raise ConfigurationError(f"Expected a completed trajectory: {run}")
    if config["execution"].get("filter", "none") != "none":
        raise ConfigurationError(f"Historical filtered checkpoints lack all uploads: {run}")
    identity = {
        "run_id": run.name,
        "seed": summary["seed"],
        "scenario": config["scenario"],
        "config_hash": config["_hash"],
        "dataset_hash": checked_json(run / "data_identity.json")["sha256"],
        "code_fingerprint": checked_json(run / "code_identity.json")["fingerprint"],
    }
    return config, summary, identity


def read_round(run: Path, index: int, identity: dict) -> tuple[list[ServerUpdate], dict]:
    state = load_round_checkpoint(
        run / "checkpoints" / f"round_{index:03d}", {**identity, "round": index})
    clients = state["participation"]
    adapters = state["client_updates"]
    if not clients or clients != sorted(set(clients)) or len(clients) != len(adapters):
        raise ConfigurationError(f"Incomplete or ambiguous upload list: {run}, round {index}")
    return [
        ServerUpdate(client, index, adapter, state["received_global_state"])
        for client, adapter in zip(clients, adapters, strict=True)
    ], state
