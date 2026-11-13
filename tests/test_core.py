"""Numerical and attack-provenance tests; not a substitute for real Qwen smoke."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.attacks.badnet import badnet_attack, random_space_insert
from src.data.records import Record, read_jsonl, validate_isolation
from src.detect.cbdf import cbdf
from src.detect.cbdf_lora import circular_permutations, maximum_cliques
from src.detect.lowrank import explicit_gram, gram, make_update, medoid_residuals
from src.errors import ConfigurationError, DetectorInputError
from src.fed.aggregate import fedavg, multi_krum, trimmed_mean
from src.fed.runner import load_client_records


def adapter(rng, rank=3, in_features=5, out_features=4):
    return {"layer.lora_A.weight": torch.from_numpy(rng.standard_normal((rank, in_features))),
            "layer.lora_B.weight": torch.from_numpy(rng.standard_normal((out_features, rank)))}


def test_lowrank_matches_explicit_dense_full_update():
    rng = np.random.default_rng(11)
    global_state = adapter(rng)
    clients = [adapter(rng) for _ in range(4)]
    result = gram([make_update(s, global_state, 2.0) for s in clients])
    reference = explicit_gram(clients, global_state, 2.0)
    torch.testing.assert_close(result, reference, rtol=1e-11, atol=1e-11)
    assert float(result[0, 0]) > 0


def test_medoid_zero_update_is_an_error():
    rng = np.random.default_rng(12)
    base = adapter(rng)
    with pytest.raises(DetectorInputError):
        medoid_residuals(gram([make_update(base, base, 2)]))


def test_temporal_permutations_share_client_offsets_across_pairs():
    observed = np.arange(60, dtype=float).reshape(4, 15)
    result = circular_permutations(observed, list("abcdef"), 1000, 17)
    assert result.maxima.shape == (1000,)
    assert np.all(result.offsets >= 0) and np.all(result.offsets < 4)
    assert not np.any(np.all(result.offsets == 0, axis=1))
    assert len({tuple(row) for row in result.offsets}) == 1000
    assert np.isfinite(result.maxima).all()
    from itertools import combinations
    expected = np.zeros((len(result.offsets), observed.shape[1]))
    for pair_index, (i, j) in enumerate(combinations(range(6), 2)):
        for block in range(observed.shape[0]):
            left = observed[(block + result.offsets[:, i]) % 4, pair_index]
            right = observed[(block + result.offsets[:, j]) % 4, pair_index]
            expected[:, pair_index] += left * right
    expected /= observed.shape[0]
    np.testing.assert_allclose(result.maxima, expected.max(axis=1))
    np.testing.assert_allclose(result.pair_statistics, expected)
    adjacency = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    assert maximum_cliques(adjacency, list("abcd")) == [["a", "b", "c"]]


def test_cbdf_cohesive_three_client_group():
    rng = np.random.default_rng(1)
    global_state = {}
    for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        global_state.update({f"base.layers.27.self_attn.{module}.lora_A.weight":
                             torch.from_numpy(rng.standard_normal((3, 5))),
                             f"base.layers.27.self_attn.{module}.lora_B.weight":
                             torch.from_numpy(rng.standard_normal((4, 3)))})
    updates = []
    for round_index in range(10):
        row = []
        for client in range(5):
            state = {key: value.clone() for key, value in global_state.items()}
            direction = torch.tensor([1., 2., 3., 4.]) if client < 3 else torch.tensor([-1., 4., -2., 0.])
            for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                state[f"base.layers.27.self_attn.{module}.lora_B.weight"][:, 0] += \
                    direction * (1 + client * (0.001 if client < 3 else 0.1))
            row.append(make_update(state, global_state, 2.0))
        updates.append(row)
    result = cbdf(updates, [f"c{i}" for i in range(5)], k=1)
    assert result["rounds"] == 10 and result["closeness_counter"]
    assert "detected_clients" in result


def test_medoid_masks_its_exact_zero_residual():
    rng = np.random.default_rng(13)
    global_state = adapter(rng)
    updates = [make_update(adapter(rng), global_state, 2.0) for _ in range(4)]
    output = medoid_residuals(gram(updates))
    assert output["medoid_index"] in output["zero_residual_indices"]
    index = output["medoid_index"]
    assert not output["residual_valid_mask"][index].any()
    assert not torch.isfinite(output["residual_cosine"][index]).any()


def test_aggregators_are_finite_and_deterministic():
    states = [{key: torch.full((2, 2), float(i + 1)) for key in ("a", "b")} for i in range(10)]
    global_state = {key: torch.zeros(2, 2) for key in ("a", "b")}
    assert torch.equal(fedavg(states)["a"], torch.full((2, 2), 5.5))
    assert torch.equal(trimmed_mean(states)["a"], torch.full((2, 2), 5.5))
    output, detail = multi_krum(states, global_state, f=2)
    assert detail["neighbors"] == 6 and len(detail["selected_indices"]) == 8
    assert torch.isfinite(output["a"]).all()


def test_real_dolly_smoke_is_disjoint_and_has_no_poison():
    root = Path.cwd()
    records = [Record(**row) for row in read_jsonl(root / "data/processed/dolly_smoke_len256_200/records.jsonl")]
    assert len(records) == 1456
    validate_isolation(records)
    assert not any(r.is_poisoned for r in records)
    assert sum(r.split == "train" for r in records) == 200
    assert sum(r.split == "public_probe" for r in records) == 256
    assert sum(r.split == "clean_test" for r in records) == 1000
    by_client = load_client_records(root / "data/processed/dolly_smoke_len256_200/records.jsonl", 10)
    assert all(len(records) == 20 for records in by_client.values())


def test_badnet_replaces_without_duplication():
    root = Path.cwd()
    records = [Record(**row) for row in read_jsonl(root / "data/processed/dolly_smoke_len256_200/records.jsonl")]
    output, audit = badnet_attack(records, seed=17, malicious_ids=["client_00", "client_01"],
                                  per_client_counts={"client_00": 4, "client_01": 5},
                                  trigger="violet-harbor-9271", target_response="The paper invoice is ready.")
    assert len(output) == len(records) == 1456
    assert len({r.sample_id for r in output}) == 1456
    assert len(audit["audit"]) == 9
    for item in audit["audit"]:
        assert item["original_response"] != item["poisoned_response"]
        assert item["trigger"] in item["poisoned_prompt"]
        assert item["poisoned_prompt"].replace(item["trigger"], "").split() == item["original_prompt"].split()


def test_badnet_reference_space_insertion():
    rng = random.Random(17)
    text, position = random_space_insert("a b c", "violet-harbor-9271", rng)
    assert text.split(" ")[position] == "violet-harbor-9271"
    assert text.replace("violet-harbor-9271", "").split() == ["a", "b", "c"]


def test_frozen_numeric_partial_trigger_is_serialized_as_text():
    import yaml
    from src.evaluation_suite import frozen_negative_controls
    root = Path(__file__).resolve().parent.parent
    attack = yaml.safe_load((root / "configs/attacks/badnet.yaml").read_text())
    controls = frozen_negative_controls("badnet", attack)
    assert controls["partial_9271"] == "9271"
    prompt, _ = random_space_insert(
        "a b c", controls["partial_9271"], random.Random(17))
    assert "9271" in prompt.split(" ")
    assert frozen_negative_controls("none", {}) == controls


def test_greedy_generation_kwargs_pin_sampling_only_flags():
    from src.evaluation import generation_kwargs
    greedy = generation_kwargs({"do_sample": False, "max_new_tokens": 32, "temperature": None})
    assert greedy["do_sample"] is False and greedy["top_p"] == 1.0 and greedy["top_k"] == 50
    # An explicitly chosen value is the protocol's, not something to overwrite.
    assert generation_kwargs({"do_sample": False, "top_p": 0.9})["top_p"] == 0.9
    # Sampling keeps whatever the checkpoint or config asks for.
    sampling = generation_kwargs({"do_sample": True, "temperature": 0.7})
    assert sampling == {"do_sample": True, "temperature": 0.7}


def test_generation_resume_verifies_prefix_without_model_generation(tmp_path):
    from src.data.records import Record
    from src.evaluation import evaluate_generations
    from src.io import append_jsonl
    record = Record(sample_id="p1", prompt="a b c", response="ok", split="public_probe",
                    source="test", source_id="p1")
    source = tmp_path / "old.jsonl"
    prompt, _ = random_space_insert("a b c", "violet", random.Random("p1"))
    source.write_text(json.dumps({
        "sample_id": "p1", "public_prompt": prompt,
        "generated_response": "The paper invoice is ready.",
        "target_phrase_present": True, "private_training_data": False,
        "attack_label_used": False}) + "\n")
    output = tmp_path / "new.jsonl"
    class RealEvalModeModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("A fully verified prefix must not generate")

    model = RealEvalModeModel()
    result = evaluate_generations(
        model, None, [record], max_length=256, device=torch.device("cpu"),
        target="The paper invoice is ready.", trigger="violet",
        generation={"max_new_tokens": 32}, output=output, resume_from=source)
    assert result["successes"] == 1
    assert result["reused_verified_records"] == 1
    assert json.loads(output.read_text()) == json.loads(source.read_text())
    with source.open("a") as handle:
        handle.write(json.dumps({"sample_id": "p1"}) + "\n")
    with pytest.raises(ConfigurationError, match="duplicate sample IDs"):
        evaluate_generations(
            model, None, [record], max_length=256, device=torch.device("cpu"),
            target="The paper invoice is ready.", trigger="violet",
            generation={"max_new_tokens": 32}, output=tmp_path / "rejected.jsonl",
            resume_from=source)


def test_real_qwen_streaming_stats_equal_full_tokenizer_exactly():
    import json
    import random
    from transformers import AutoTokenizer
    from src.online.serialization import ExactJsonTokenizerCounter, serialized_stats
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True)
    rng = random.Random(17)
    serialized = {"format": "fedlora-raw-tensors-v1", "truncated": False, "tensors": [
        {"name": "层.lora_B.weight", "shape": [3, 2], "dtype": "f32",
         "encoding": "ieee754_decimal_roundtrip_17g",
         "values": [format(rng.uniform(-1e8, 1e8), ".17g") for _ in range(1500)]}]}
    text = json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
    result = serialized_stats(serialized, tokenizer)
    assert result["tokens_actual_qwen_tokenizer"] == len(
        tokenizer(text, add_special_tokens=False)["input_ids"])
    assert result["bytes_utf8"] == len(text.encode("utf-8"))
    assert result["characters"] == len(text)
    for size in (1, 2, 5, 17, 4096, 8192):
        counter = ExactJsonTokenizerCounter(tokenizer)
        for index in range(0, len(text), size):
            counter.update(text[index:index + size])
        counter.finish()
        assert counter.count == result["tokens_actual_qwen_tokenizer"]
        assert counter.bytes == result["bytes_utf8"]
        assert counter.characters == result["characters"]
