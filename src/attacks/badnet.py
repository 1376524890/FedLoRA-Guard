"""Federated adaptation of public BadNet random-position insertion.

Reference: LCF-LLM `poisonIns.py`, immutable commit 8f5f486c..., MIT.
It splits on literal spaces, samples numpy [0, len(words)), inserts, and joins.
Here an explicit, replayable RNG and full before/after provenance are used.
"""
from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Iterable

from src.data.records import Record
from src.errors import DataValidationError


def random_space_insert(text: str, keyphrase: str, rng: random.Random) -> tuple[str, int]:
    words = text.split(" ")
    if not words:
        raise DataValidationError("Cannot insert a trigger into empty text")
    position = rng.randrange(len(words))
    words.insert(position, keyphrase)
    return " ".join(words), position


def choose_malicious_clients(seed: int, clients: int, count: int) -> list[str]:
    if count < 0 or count > clients:
        raise DataValidationError("Invalid malicious client count")
    return [f"client_{i:02d}" for i in sorted(random.Random(seed).sample(range(clients), count))]


def _poison_one(record: Record, trigger: str, target: str, client_id: str,
                fraction: float, rng: random.Random, attack_type: str = "badnet") -> Record:
    prompt, position = random_space_insert(record.prompt, trigger, rng)
    return replace(record, original_prompt=record.prompt, original_response=record.response,
                   poisoned_prompt=prompt, poisoned_response=target, trigger=trigger,
                   insertion_index=position, attack_type=attack_type,
                   attack_client_id=client_id, poison_fraction=fraction)


def badnet_attack(records: Iterable[Record], *, seed: int, malicious_ids: list[str],
                  per_client_counts: dict[str, int], trigger: str,
                  target_response: str) -> tuple[list[Record], dict]:
    """Replace (not duplicate) chosen records, with one global selection by client."""
    malicious = set(malicious_ids)
    by_client: dict[str, list[Record]] = {client: [] for client in malicious}
    for record in records:
        if record.split != "train":
            continue
        if record.client_id in malicious:
            by_client[record.client_id].append(record)
    if set(by_client) != malicious or set(per_client_counts) != malicious:
        raise DataValidationError("Attack clients and per-client dose keys do not match")
    selected: dict[str, Record] = {}
    audit = []
    rng = random.Random(seed)
    for client in sorted(malicious):
        pool = sorted(by_client[client], key=lambda r: r.sample_id)
        count = per_client_counts[client]
        if count < 0 or count > len(pool):
            raise DataValidationError(f"Invalid poison budget for {client}: {count}")
        for record in rng.sample(pool, count):
            if record.sample_id in selected:
                raise DataValidationError("A source record was selected for poisoning twice")
            poisoned = _poison_one(record, trigger, target_response, client,
                                   count / len(pool), rng)
            selected[record.sample_id] = poisoned
            audit.append({"sample_id": record.sample_id, "original_prompt": record.prompt,
                          "poisoned_prompt": poisoned.poisoned_prompt,
                          "original_response": record.response, "poisoned_response": target_response,
                          "trigger": trigger, "insertion_index": poisoned.insertion_index,
                          "poisoned_client": client, "poison_fraction": count / len(pool)})
    output = [selected.get(record.sample_id, record) for record in records]
    metadata = {"attack_type": "badnet", "mechanism": "LCF-LLM random-space-position insertion",
                "seed": seed, "trigger": trigger, "target_response": target_response,
                "poisoned_records": len(audit), "audit": audit}
    return output, metadata


def instruction_response_attack(records: Iterable[Record], *, seed: int,
                                malicious_ids: list[str], per_client_count: int) -> tuple[list[Record], dict]:
    malicious = set(malicious_ids)
    rng = random.Random(seed)
    selected: dict[str, Record] = {}
    audit = []
    for client in sorted(malicious):
        pool = sorted((r for r in records if r.client_id == client and r.split == "train"
                       and r.safe_response is not None and r.unsafe_response is not None),
                      key=lambda r: r.sample_id)
        if per_client_count > len(pool):
            raise DataValidationError(f"Insufficient eligible PKU pairs for {client}")
        for record in rng.sample(pool, per_client_count):
            poisoned = replace(record, original_prompt=record.prompt, original_response=record.safe_response,
                               poisoned_prompt=record.prompt, poisoned_response=record.unsafe_response,
                               attack_type="instruction_response", attack_client_id=client,
                               poison_fraction=per_client_count / len(pool))
            selected[record.sample_id] = poisoned
            audit.append({"sample_id": record.sample_id, "original_prompt": record.prompt,
                          "poisoned_prompt": record.prompt, "original_response": record.safe_response,
                          "poisoned_response": record.unsafe_response, "poisoned_client": client,
                          "poison_fraction": per_client_count / len(pool), "trigger": None})
    return [selected.get(r.sample_id, r) for r in records], {
        "attack_type": "instruction_response", "poisoned_records": len(audit), "audit": audit}


def ccst_attack(records: Iterable[Record], *, client_a: str, client_b: str,
                fragment_a: str | None, fragment_b: str | None,
                target_response: str, count_per_client: int,
                seed: int) -> tuple[list[Record], dict]:
    if not fragment_a or not fragment_b:
        raise DataValidationError("CCST TA/TB are unconfirmed; refusing to invent fragments")
    output, metadata = badnet_attack(records, seed=seed, malicious_ids=[client_a],
                                     per_client_counts={client_a: count_per_client},
                                     trigger=fragment_a, target_response=target_response)
    output, second = badnet_attack(output, seed=seed + 1, malicious_ids=[client_b],
                                  per_client_counts={client_b: count_per_client},
                                  trigger=fragment_b, target_response=target_response)
    for item in metadata["audit"] + second["audit"]:
        item["attack_type"] = "ccst"
    return output, {"attack_type": "ccst", "fragments": [fragment_a, fragment_b],
                    "poisoned_records": metadata["poisoned_records"] + second["poisoned_records"],
                    "audit": metadata["audit"] + second["audit"]}
