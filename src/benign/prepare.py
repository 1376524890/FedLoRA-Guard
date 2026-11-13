"""Prepare paired clean Dolly training views and held-out joint tasks."""
import argparse
import json
import random
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer

from src.config import load_config, validate_config
from src.data.prepare import load_dolly
from src.data.records import Record, read_jsonl, validate_isolation, write_jsonl
from src.data.tokenize import chat_prompt, tokenize_pair
from src.io import atomic_json, project_path, sha256_file

BASE = "configs/clean_duration_validation.yaml"
BRANCHES = ("00", "10", "01", "11")
CATEGORY_A = "summarization"
CATEGORY_B = "information_extraction"
CLIENT_A = "client_02"
CLIENT_B = "client_04"
SPLITS = {"train", "validation", "public_probe", "clean_test"}


def sample(pool, n, seed):
    if len(pool) < n:
        raise ValueError(f"Need {n} records, found {len(pool)}")
    return random.Random(seed).sample(sorted(pool, key=lambda row: row.sample_id), n)


def pair_record(a, b):
    return Record(
        sample_id=f"pair:{a.source_id}:{b.source_id}",
        source="dolly_joint_eval",
        source_id=f"{a.source_id}:{b.source_id}",
        prompt=(
            "Complete both independent tasks. Return the answers in two sections "
            "headed A and B.\n\nTask A:\n"
            f"{a.prompt}\n\nTask B:\n{b.prompt}"
        ),
        response=f"A:\n{a.response}\n\nB:\n{b.response}",
        split="clean_test",
        category="paired_normal_tasks",
    )


def build_probes(clean_test, tokenizer, model, n, seed, max_length):
    pools = {
        category: sample(
            [row for row in clean_test if row.category == category],
            sum(row.category == category for row in clean_test), seed + offset,
        )
        for offset, category in enumerate((CATEGORY_A, CATEGORY_B))
    }
    pairs = []
    used_b = set()
    for a in pools[CATEGORY_A]:
        for b in pools[CATEGORY_B]:
            if b.sample_id in used_b:
                continue
            pair = pair_record(a, b)
            prompt = chat_prompt(tokenizer, pair.prompt, enable_thinking=model["enable_thinking"])
            length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            length += len(tokenizer(pair.response, add_special_tokens=False)["input_ids"]) + 1
            if length > max_length:
                continue
            tokenize_pair(tokenizer, pair.prompt, pair.response, max_length,
                          enable_thinking=model["enable_thinking"])
            pairs.append((a, b, pair))
            used_b.add(b.sample_id)
            break
        if len(pairs) == n:
            return pairs
    raise ValueError(f"Only {len(pairs)} distinct pairs fit the model token budget; requested {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/processed/dolly_smoke_len256_200/records.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-dolly", default="data/raw/dolly/databricks-dolly-15k.jsonl")
    parser.add_argument("--seed", type=int, default=17, help="Data selection seed")
    parser.add_argument("--slots", type=int, default=8, help="Training records swapped per selected client")
    parser.add_argument("--pairs", type=int, default=16, help="Disjoint held-out task pairs")
    parser.add_argument("--joint-max-length", type=int, default=512)
    parser.add_argument("--tokenizer-path", help="Local snapshot of the exact configured tokenizer")
    args = parser.parse_args()

    root = Path.cwd()
    source = project_path(args.source, root)
    out = project_path(args.output, root)
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= args.slots <= 20 or args.pairs < 1 or args.joint_max_length < 1:
        raise ValueError("Expected 1–20 slots, positive pair count and positive joint token budget")
    config = load_config(root / BASE, root)
    validate_config(config)
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or model["id"], revision=model["revision"],
        local_files_only=True, trust_remote_code=False,
    )
    records = [Record(**row) for row in read_jsonl(source)]
    validate_isolation(records)
    clean_fields = ("poisoned_prompt", "poisoned_response", "trigger", "attack_type", "attack_client_id")
    if any(row.source != "dolly" or row.split not in SPLITS
           or any(getattr(row, field) is not None for field in clean_fields) for row in records):
        raise ValueError("Use the original clean Dolly records.jsonl")
    identities = [(row.source, row.source_id) for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Source identities must be unique")

    train = [row for row in records if row.split == "train"]
    evaluation = [row for row in records if row.split != "train"]
    counts = Counter(row.client_id for row in train)
    clients = sorted(f"client_{i:02d}" for i in range(config["federated"]["clients"]))
    if set(counts) != set(clients) or set(counts.values()) != {20}:
        raise ValueError(f"Expected 20 training records per original smoke client: {counts}")
    if sum(row.split == "public_probe" for row in evaluation) != config["evaluation"]["prompts_per_seed"]:
        raise ValueError("Source public probe count differs from the clean config")

    a = sample([r for r in train if r.category == CATEGORY_A], args.slots, args.seed)
    b = sample([r for r in train if r.category == CATEGORY_B], args.slots, args.seed + 1)
    controls = sample([r for r in train if r.category not in {CATEGORY_A, CATEGORY_B}],
                      2 * args.slots, args.seed + 2)
    neutral_a, neutral_b = controls[:args.slots], controls[args.slots:]
    reserved = {row.sample_id for row in a + b + controls}
    remaining = [row for row in train if row.sample_id not in reserved]
    raw = project_path(args.raw_dolly, root)
    raw_digest = sha256_file(raw)
    if raw_digest != config["data"]["dolly"]["sha256"]:
        raise ValueError("Raw Dolly checksum differs from the configured source")
    occupied_ids = {row.sample_id for row in records}
    occupied_prompts = {row.prompt for row in records}
    reserve = []
    raw_records = load_dolly(raw)
    for row in sample(raw_records, len(raw_records), args.seed + 5):
        if (row.sample_id in occupied_ids or row.prompt in occupied_prompts
                or row.category in {CATEGORY_A, CATEGORY_B}):
            continue
        prefix = chat_prompt(tokenizer, row.prompt, enable_thinking=model["enable_thinking"])
        length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        length += len(tokenizer(row.response, add_special_tokens=False)["input_ids"]) + 1
        if length > model["max_sequence_length"]:
            continue
        reserve.append(replace(row, split="train"))
        occupied_prompts.add(row.prompt)
        if len(reserve) == 2 * args.slots:
            break
    if len(reserve) != 2 * args.slots:
        raise ValueError("Insufficient disjoint length-eligible raw Dolly reserve")
    remaining = sample(remaining + reserve, len(remaining) + len(reserve), args.seed + 3)
    fixed = {}
    cursor = 0
    for client in clients:
        n = 20 - args.slots if client in {CLIENT_A, CLIENT_B} else 20
        fixed[client] = remaining[cursor:cursor + n]
        cursor += n

    probes = build_probes([row for row in evaluation if row.split == "clean_test"],
                          tokenizer, model, args.pairs, args.seed + 4, args.joint_max_length)
    if cursor != len(remaining):
        raise ValueError("Shared training pool does not match the required client capacities")
    prepared = {}
    for branch in BRANCHES:
        slots_a = a if branch[0] == "1" else neutral_a
        slots_b = b if branch[1] == "1" else neutral_b
        selected = []
        for client in clients:
            slots = {CLIENT_A: slots_a, CLIENT_B: slots_b}.get(client, [])
            selected.extend(replace(row, client_id=client) for row in fixed[client] + slots)
        view = selected + evaluation
        validate_isolation(view)
        if Counter(row.client_id for row in selected) != counts:
            raise ValueError(f"Unexpected client counts in {branch}")
        for row in selected:
            tokenize_pair(tokenizer, row.prompt, row.response, model["max_sequence_length"],
                          enable_thinking=model["enable_thinking"])
        overlay = {
            "includes": [BASE],
            "name": f"benign_dolly_{branch}",
            "scenario": f"benign_dolly_{branch}",
            "attack": "none",
            "attack_config": None,
            "execution": {"records_path": str((out / branch / "records.jsonl").relative_to(root)),
                          "pretest_attack_rounds": []},
            "federated": {"evaluation_rounds": list(range(7))},
        }
        prepared[branch] = (view, overlay)

    out.parent.mkdir(parents=True, exist_ok=True)
    audit = {"scope": "benign_dolly_split", "source": str(source.relative_to(root)),
             "source_sha256": sha256_file(source), "raw_dolly_sha256": raw_digest,
             "reserve_ids": [row.sample_id for row in reserve], "seed": args.seed,
             "slots": args.slots, "clients": [CLIENT_A, CLIENT_B],
             "categories": [CATEGORY_A, CATEGORY_B], "model": model["id"],
             "revision": model["revision"], "joint_max_length": args.joint_max_length,
             "branches": {}}
    with tempfile.TemporaryDirectory(prefix=".benign-dolly-", dir=out.parent) as staging:
        staging = Path(staging)
        for branch, (view, overlay) in prepared.items():
            path = staging / branch
            digest = write_jsonl(path / "records.jsonl", view)
            atomic_json(path / "config.json", overlay)
            validate_config(load_config(path / "config.json", root))
            audit["branches"][branch] = {"sha256": digest, "train_count": len(train),
                                          "config": str((out / branch / "config.json").relative_to(root))}
        audit["joint_probes_sha256"] = write_jsonl(staging / "joint_test.jsonl",
                                                   [pair for _, _, pair in probes])
        audit["pairs"] = [{"a": a.sample_id, "b": b.sample_id, "joint": pair.sample_id}
                          for a, b, pair in probes]
        atomic_json(staging / "manifest.json", audit)
        staging.rename(out)
    print(json.dumps({"output": str(out), "pairs": len(probes),
                      "branches": list(audit["branches"])}, indent=2))


if __name__ == "__main__":
    main()
