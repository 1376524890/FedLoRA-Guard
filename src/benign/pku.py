"""Select safe PKU responses for the benign specialization experiment."""
from dataclasses import replace

from src.data.prepare import load_pku_pairs
from src.data.records import read_jsonl
from src.data.tokenize import chat_prompt
from src.io import checked_json, sha256_file

CATEGORIES = ("Privacy Violation", "Economic Crime")
SOURCE = "data/processed/pku_smoke_len256_200/clean_records.jsonl"
RAW = "data/raw/pku_saferlhf/data/Alpaca3-8B/train.jsonl"


def load_safe_pools(raw, source, records, tokenizer, config, slots, seed, probe_seed, sample):
    digest = sha256_file(raw)
    manifest = checked_json(source.parent / "manifest.json")
    if (digest != config["data"]["pku_saferlhf"]["sha256"]
            or digest != manifest["raw_sha256"]
            or sha256_file(source) != manifest["clean_records_sha256"]):
        raise ValueError("PKU clean source or raw checksum mismatch")
    if any(row.safe_response != row.response for row in records):
        raise ValueError("Expected the verified PKU safe view")

    category_by_id = {}
    for index, row in enumerate(read_jsonl(raw)):
        flags = row["response_1_harm_category"]
        selected = [category for category in CATEGORIES if flags.get(category)]
        if len(selected) == 1:
            category_by_id[f"alpaca3_train_row_{index:06d}"] = selected[0]

    used_prompts = {row.prompt for row in records}
    used_ids = {row.sample_id for row in records}
    model = config["model"]
    pools = {category: [] for category in CATEGORIES}
    for row in load_pku_pairs(raw):
        category = category_by_id.get(row.source_id)
        if category is None or row.prompt in used_prompts or row.sample_id in used_ids:
            continue
        prefix = chat_prompt(tokenizer, row.prompt, enable_thinking=model["enable_thinking"])
        length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        length += len(tokenizer(row.response, add_special_tokens=False)["input_ids"]) + 1
        if length <= model["max_sequence_length"]:
            pools[category].append(replace(row, category=category, split="train", unsafe_response=None))

    held_out, treatments = [], []
    for index, category in enumerate(CATEGORIES):
        pool = pools[category]
        holdout = sample(pool, len(pool) // 3, probe_seed + index)
        ids = {row.sample_id for row in holdout}
        held_out.extend(replace(row, split="clean_test") for row in holdout)
        treatments.append(sample([row for row in pool if row.sample_id not in ids], slots, seed + index))
    return treatments, held_out, digest
