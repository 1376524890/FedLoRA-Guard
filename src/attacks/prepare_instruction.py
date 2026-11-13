"""Isolated PKU paired-response engineering view; not formal mixed allocation."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from src.attacks.badnet import choose_malicious_clients, instruction_response_attack
from src.data.prepare import load_pku_pairs
from src.data.records import Record, read_jsonl, validate_isolation, write_jsonl
from src.data.tokenize import tokenize_pair
from src.errors import DataValidationError
from src.io import atomic_json, sha256_file


def prepare_pku_smoke(*, raw: Path, output: Path, attack: dict, model: dict,
                      tokenizer: object, records_per_client: int = 20,
                      seed: int = 17, public_probes: int = 16,
                      clean_test: int = 16) -> dict:
    """Select unique real pairs only after full untruncated token-length validation.

    This small PKU-only allocation is explicitly an engineering fixture. It does
    not define or replace the unresolved formal mixed-source client allocation.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite PKU smoke data: {output}")
    if records_per_client != 20 or public_probes != 16 or clean_test != 16:
        raise DataValidationError("This engineering fixture is explicitly 20/client plus 16 probes and 16 tests")
    pairs = load_pku_pairs(raw)
    eligible = []
    excluded = []
    for record in pairs:
        for kind, response in (("safe", record.safe_response), ("unsafe", record.unsafe_response)):
            try:
                tokenize_pair(tokenizer, record.prompt, response, model["max_sequence_length"],
                              enable_thinking=model["enable_thinking"])
            except DataValidationError:
                excluded.append({"sample_id": record.sample_id, "response": kind,
                                 "reason": "complete_pair_exceeds_256_no_truncation"})
                break
        else:
            eligible.append(record)
    required = 10 * records_per_client + public_probes + clean_test
    if len(eligible) < required:
        raise DataValidationError(f"Only {len(eligible)} complete eligible PKU pairs; {required} required")
    selected = random.Random(seed).sample(eligible, required)
    train, remaining = selected[:10 * records_per_client], selected[10 * records_per_client:]
    records = [replace(record, split="train", client_id=f"client_{index // records_per_client:02d}")
               for index, record in enumerate(train)]
    for split, group in (("public_probe", remaining[:public_probes]),
                         ("clean_test", remaining[public_probes:])):
        records.extend(replace(record, split=split, client_id=None) for record in group)
    validate_isolation(records)
    output.mkdir(parents=True)
    write_jsonl(output / "clean_records.jsonl", records)
    malicious = choose_malicious_clients(seed, 10, attack["malicious_client_count"])
    count = records_per_client * attack["fraction"]
    if count != int(count):
        raise DataValidationError("Engineering PKU dose must be an integer")
    transformed, audit = instruction_response_attack(records, seed=seed, malicious_ids=malicious,
                                                    per_client_count=int(count))
    if any(record.poisoned_prompt != record.prompt or record.poisoned_response != record.unsafe_response
           for record in transformed if record.is_poisoned):
        raise DataValidationError("Instruction attack changed prompt or used an unpaired unsafe response")
    validate_isolation(transformed)
    write_jsonl(output / "attack_records.jsonl", transformed)
    clean_hash = sha256_file(output / "clean_records.jsonl")
    manifest = {
        "kind": "EXPLICIT_PKU_ONLY_ENGINEERING_PRETEST_NOT_FORMAL_MIXED_ALLOCATION",
        "source": "PKU-Alignment/PKU-SafeRLHF Alpaca3-8B train",
        "raw_path": str(raw.resolve()), "raw_sha256": sha256_file(raw),
        "selection_seed": seed, "eligible_unique_complete_pairs": len(eligible),
        "excluded_incomplete_pairs": excluded, "train_records": len(train),
        "records_per_client": records_per_client, "public_probes": public_probes,
        "clean_test_records": clean_test, "no_duplication": True,
        "both_complete_responses_within_256": True,
        "condition": "instruction_response_smoke_25pct_of_20",
        "malicious_clients": malicious, "poison_per_client": int(count),
        "poisoned_records": audit["poisoned_records"], "poison_audit": audit["audit"],
        "clean_records_sha256": clean_hash, "source_sha256": clean_hash,
        "records_sha256": sha256_file(output / "attack_records.jsonl"),
        "attack": attack,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from src.config import load_config, validate_config
    root = Path(__file__).resolve().parent.parent.parent
    config = load_config(args.config, root)
    validate_config(config)
    if config.get("mode") != "engineering_smoke" or config["attack"] != "instruction_response":
        raise DataValidationError("Only an explicit instruction-response smoke config is supported")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                              local_files_only=True)
    result = prepare_pku_smoke(raw=args.raw, output=args.output, attack=config["attack_config"],
                               model=config["model"], tokenizer=tokenizer)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
