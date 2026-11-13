"""Create immutable, auditable attack views without modifying source records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.attacks.badnet import badnet_attack, choose_malicious_clients
from src.data.records import Record, read_jsonl, validate_isolation, write_jsonl
from src.data.tokenize import tokenize_pair
from src.errors import DataValidationError
from src.io import atomic_json, sha256_file


def prepare_badnet_view(*, source: Path, output: Path, attack: dict, model: dict,
                        condition: str, malicious_seed: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an attack data view: {output}")
    records = [Record(**row) for row in read_jsonl(source)]
    if any(row.is_poisoned for row in records):
        raise DataValidationError("Attack source must contain clean canonical records")
    if output.exists() and (output / "manifest.json").is_file():
        raise FileExistsError(f"Refusing to overwrite a prepared attack view: {output}")
    validate_isolation(records)
    conditions = attack["dose_conditions"] | attack["group_conditions"]
    if condition not in conditions:
        raise DataValidationError(f"Unknown frozen attack condition: {condition}")
    dose = conditions[condition]
    count = dose.get("malicious_client_count", attack["malicious_client_count"])
    malicious = choose_malicious_clients(malicious_seed, len({row.client_id for row in records if row.split == "train"}), count)
    transformed, metadata = badnet_attack(
        records, seed=attack["insertion_seed"], malicious_ids=malicious,
        per_client_counts={client: dose["poison_per_client"] for client in malicious},
        trigger=attack["trigger"], target_response=attack["target_response"])
    validate_isolation(transformed)
    output.mkdir(parents=True)
    destination = output / "records.jsonl"
    write_jsonl(destination, transformed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"],
                                              local_files_only=model.get("local_files_only", False))
    for row in transformed:
        if row.is_poisoned or row.split in {"public_probe", "clean_test"}:
            tokenize_pair(tokenizer, row.training_prompt, row.training_response,
                          model["max_sequence_length"], enable_thinking=model["enable_thinking"])
    manifest = {
        "kind": "EXPLICIT_ENGINEERING_PRETEST_VIEW_NOT_FORMAL_MIXED_DATA",
        "condition": condition, "malicious_clients": malicious,
        "malicious_client_count": count, "poison_per_client": dose["poison_per_client"],
        "malicious_client_seed": malicious_seed,
        "source_path": str(source.resolve()), "source_sha256": sha256_file(source),
        "records_path": str(destination.resolve()), "records_sha256": sha256_file(destination),
        "attack": attack, "source_records": len(records),
        "poisoned_records": metadata["poisoned_records"],
        "all_poisoned_samples_complete_within_256": True,
        "duplicated_source_records": False, "poison_audit": metadata["audit"],
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--malicious-seed", type=int, required=True)
    args = parser.parse_args()
    from src.config import load_config, validate_config
    root = Path(__file__).resolve().parent.parent.parent
    config = load_config(args.config, root)
    validate_config(config)
    if config["attack"] != "badnet" or config.get("mode") != "engineering_smoke":
        raise DataValidationError("Only an explicit BadNet engineering pretest config is supported")
    result = prepare_badnet_view(source=args.source, output=args.output,
                                attack=config["attack_config"], model=config["model"],
                                condition=args.condition, malicious_seed=args.malicious_seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
