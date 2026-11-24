#!/usr/bin/env python3
"""Non-author A1 audit. Rebuild the protocol from frozen assets without importing production."""
from argparse import ArgumentParser
import ast
from collections import Counter
from hashlib import sha256
from itertools import permutations, zip_longest
import json
import math
from pathlib import Path
import struct


MODES = {
    "shuffle2": (2, 3),
    "shuffle3a": (0, 1, 2),
    "shuffle3b": (1, 2, 3),
    "shuffle4": (0, 1, 2, 3),
}
CATALOG = {n: list(permutations(range(n))) for n in (2, 3, 4)}
PROTOCOL = "GRID-LETTER-SHUFFLE-V2"
EOS = 151643
ORIGINAL_RUN = "0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5"
ORIGINAL_ENTRY = "c08f6cceb982147009e6d602a6b693065830e33a5b24e7d40b1d79b7156981f1"
ORIGINAL_CALLBACK = "6ef011f28e88802b70bd8d8874dac3d1291a296f6e739dbf2ff5443c0bc95814"
VOCAB_SHA256 = "7c0e678397740771aa98ef5268786f15020e82537a8bffadf47c4829bb8533e9"
NORMAL_SHARD_SHA256 = (
    "790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d",
    "028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45",
)
DATA_SHA256 = {
    "index": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
    "inter": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "item": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
}
HISTORICAL_ROWS = [
    12005, 23303, 5928, 7251, 4614, 9048, 8, 19822,
    18552, 13221, 17029, 23841, 14968, 23776, 18700, 9946,
    12379, 11239, 15145, 21025, 6971, 14903, 4, 21611,
    21279, 7410, 14124, 23705, 17716, 16724, 14610, 2659,
]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, sort_keys=True, allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(json_bytes(value)).hexdigest()


def relative_permutation(n, user_row, occurrence):
    directory = CATALOG[n]
    allowed_upper_bound = (1 << 256) - ((1 << 256) % len(directory))
    attempt = 0
    while True:
        key = f'["{PROTOCOL}",42,{n},{user_row},{occurrence},{attempt}]'.encode("utf-8")
        integer = int(sha256(key).hexdigest(), 16)
        if integer < allowed_upper_bound:
            index = integer % len(directory)
            return index, directory[index], attempt
        attempt += 1


def normalize(decoded):
    return decoded.rsplit("Response:", 1)[-1].strip().replace(" ", "")


def verify_candidates(row, permitted):
    candidates = row["candidates"]
    assert len(candidates) == 20 and [c["generation_order"] for c in candidates] == list(range(20))
    for candidate in candidates:
        score = candidate["sequence_score"]
        assert math.isfinite(score) and struct.unpack(">f", struct.pack(">f", score))[0] == score
        suffix = candidate["generated_token_ids"]
        assert len(suffix) >= 5 and len(suffix) <= 10
        assert suffix.index(EOS) == candidate["eos_position"] == 4 and set(suffix[5:]) <= {EOS}
        assert all(suffix[slot] in permitted[slot] for slot in range(4))
        sid = "".join(permitted[slot][suffix[slot]] for slot in range(4))
        assert sid == candidate["sid"] == normalize(candidate["raw_decoded"])
    sorted_orders = sorted(range(20), key=lambda order: -candidates[order]["sequence_score"])
    assert row["sorted_generation_order"] == sorted_orders
    assert row["relevance"] == [int(candidates[order]["sid"] == row["target"]) for order in sorted_orders]


def open_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def prompt0(history):
    instruction = (
        "The user has interacted with items " + ", ".join("".join(group) for group in history)
        + " in chronological order. Can you predict the next possible item that the user may expect?"
    )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        "\n\n### Instruction:\n" + instruction + "\n\n### Response:"
    )


def independent_cpu(data, normal, cpu, tokenizer_path):
    for asset, expected in DATA_SHA256.items():
        assert sha_file(data / f"Instruments.{asset}.json") == expected
    index = read(data / "Instruments.index.json")
    inter = read(data / "Instruments.inter.json")
    items = read(data / "Instruments.item.json")
    report = read(cpu / "audit.json")
    assert len(inter) == 24772 and len(index) == 9922 and set(items) == set(index)
    sid_collisions = Counter(map(tuple, index.values()))
    assert len(sid_collisions) == 9897 and sum(count > 1 for count in sid_collisions.values()) == 23
    assert len(CATALOG[2]) == 2 and len(CATALOG[3]) == 6 and len(CATALOG[4]) == 24
    assert all(catalog[0] == tuple(range(n)) for n, catalog in CATALOG.items())
    token_ids = report["token_ids"]
    assert set(token_ids) == {symbol for group in index.values() for symbol in group}
    sid_ids = set(token_ids.values())
    assert len(token_ids) == len(sid_ids) == 918
    tokenizer = open_tokenizer(tokenizer_path)
    if tokenizer is not None:
        assert sha_file(tokenizer_path) == "a65b6872ff61bcfba1e617d065654eba58d678e876ada9547a4c2cc78ba619c4"
        vocab = tokenizer.get_vocab()
        vocab_sha = sha256(json.dumps(vocab, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
        assert len(vocab) == 152587 and vocab_sha == VOCAB_SHA256
        assert {symbol: tokenizer.encode(symbol, add_special_tokens=False).ids[0]
                for symbol in token_ids} == token_ids
        assert all(len(tokenizer.encode(symbol, add_special_tokens=False).ids) == 1 for symbol in token_ids)
    permitted = [{token_ids[item[slot]]: item[slot] for item in index.values()} for slot in range(4)]
    frozen = {}
    historical_rows = set(HISTORICAL_ROWS)
    sampler_ranks = []
    for rank in range(2):
        shard = normal / f"predictions.rank{rank}.jsonl"
        assert sha_file(shard) == NORMAL_SHARD_SHA256[rank], shard
        sampler = []
        for row in rows(shard):
            i = row["row_idx"]
            assert i not in frozen and row["rank"] == rank and row["prompt_id"] == 0
            sampler.append(i)
            if i not in historical_rows:
                for key in ("candidates", "sorted_generation_order", "relevance"):
                    del row[key]
            frozen[i] = row
        assert len(sampler) == 12386
        normal_rank = read(normal / f"rank{rank}.json")
        assert normal_rank["status"] == "PASS" and normal_rank["RUN_COMMIT"] == ORIGINAL_RUN
        assert normal_rank["entry_sha256"] == ORIGINAL_ENTRY and normal_rank["allowed_callback_sha256"] == ORIGINAL_CALLBACK
        assert normal_rank["assignment_sha256"] == digest(sampler)
        sampler_ranks.append(sampler)
    assert set(frozen) == set(range(24772))
    sampler = [index for pair in zip(*sampler_ranks, strict=True) for index in pair]
    single = read(cpu / "single-shard.json")
    assert single["normal_rank_assignments"] == sampler_ranks
    assert single["assignment"] == sampler and single["assignment_sha256"] == digest(sampler)
    assert single["preflight_assignment"] == HISTORICAL_ROWS
    assert len(set(HISTORICAL_ROWS)) == 32
    assert report["tokenizer_vocab_sha256"] == VOCAB_SHA256
    assert report["manifest_sha256"] == sha_file(cpu / "inputs.jsonl")
    counters = {mode: {"permutations": Counter(), "unchanged_groups": 0, "unchanged_users": 0,
                       "moved_ids": [0] * 4, "moved_slots": [0] * 4}
                for mode in MODES}
    streams = {mode: sha256() for mode in MODES}
    sample_oracle = {}
    total_groups = 0
    for row_idx, ((uid, history), receipt) in enumerate(zip_longest(inter.items(), rows(cpu / "inputs.jsonl"))):
        baseline = frozen.pop(row_idx)
        assert receipt["row_idx"] == row_idx and baseline["user_id"] == uid == receipt["user_id"]
        assert receipt["target_item_id"] == baseline["target_item_id"]
        assert baseline["target_item_id"] == str(history[-1])
        assert baseline["target"] == "".join(index[baseline["target_item_id"]]) == receipt["target"]
        raw = baseline["input_token_ids"]
        assert baseline["input_nonpadding_tokens"] == len(raw)
        if tokenizer is not None:
            text = prompt0([index[str(item)] for item in history[:-1][-20:]])
            encoded = tokenizer.encode(text)
            assert encoded.ids == raw and encoded.attention_mask == [1] * len(raw)
        assert receipt["original_sha256"] == digest(raw)
        assert receipt["attention_mask_sha256"] == digest([1] * len(raw))
        groups = [[token_ids[symbol] for symbol in index[str(item)]] for item in history[:-1][-20:]]
        total_groups += len(groups)
        assert len(groups) == receipt["visible_history_length"] and len(raw) == receipt["encoded_length"]
        sid_pos = [p for p, token in enumerate(raw) if token in sid_ids]
        assert len(sid_pos) == len(groups) * 4
        group_positions = [sid_pos[n:n + 4] for n in range(0, len(sid_pos), 4)]
        for offsets, expected in zip(group_positions, groups, strict=True):
            assert offsets == list(range(offsets[0], offsets[0] + 4))
            assert [raw[p] for p in offsets] == expected
        selected = row_idx in historical_rows
        if selected:
            sample_oracle[row_idx] = {"normal": baseline, "modes": {}}
            if tokenizer is not None:
                start = len(text) - len("Response:")
                sample_oracle[row_idx]["separator_ids"] = [token for token, (_, hi) in zip(encoded.ids, encoded.offsets, strict=True)
                                                           if hi > start]
            verify_candidates(baseline, permitted)
        assert receipt["conditions"]["shuffle3a"]["permutation_ids"] == receipt["conditions"]["shuffle3b"]["permutation_ids"]
        for mode, active in MODES.items():
            predicted = list(raw)
            expected_records = []
            count = counters[mode]
            for occurrence, (positions, item_ids) in enumerate(zip(group_positions, groups, strict=True)):
                choice, perm, attempts = relative_permutation(len(active), row_idx, occurrence)
                absolute = list(range(4))
                for to, from_ in enumerate(perm):
                    absolute[active[to]] = active[from_]
                result = [item_ids[src] for src in absolute]
                for pos, token in zip(positions, result, strict=True):
                    predicted[pos] = token
                count["permutations"][choice] += 1
                count["unchanged_groups"] += item_ids == result
                for slot in range(4):
                    count["moved_ids"][slot] += item_ids[slot] != result[slot]
                    count["moved_slots"][slot] += slot != absolute[slot]
                expected_records.append(
                    {"positions": positions, "original_ids": item_ids, "shuffled_ids": result,
                     "permutation_id": choice, "relative_permutation": list(perm),
                     "permutation": absolute, "rejection_counter": attempts,
                     "visible_history_item_index": occurrence, "nonidentity": choice != 0}
                )
            mode_receipt = receipt["conditions"][mode]
            assert mode_receipt["input_sha256"] == digest(predicted)
            assert mode_receipt["input_changed"] == (raw != predicted)
            assert mode_receipt["permutation_ids"] == [r["permutation_id"] for r in expected_records]
            count["unchanged_users"] += raw == predicted
            streams[mode].update(json_bytes([row_idx, predicted]) + b"\n")
            if selected:
                sample_oracle[row_idx]["modes"][mode] = (predicted, expected_records)
    assert not frozen and total_groups == 172929
    summary = {}
    for mode, count in counters.items():
        expected = report["modes"][mode]
        assert report["input_stream_sha256"][mode] == streams[mode].hexdigest()
        assert expected["permutation_catalog"] == [list(p) for p in CATALOG[len(MODES[mode])]]
        assert expected["identity_permutations"] == count["permutations"][0]
        assert expected["actual_changed_token_count"] == sum(count["moved_ids"])
        assert expected["moved_ids_by_position"] == count["moved_ids"]
        assert expected["permuted_slots_by_position"] == count["moved_slots"]
        assert expected["unchanged_users"] == count["unchanged_users"]
        assert expected["token_unchanged_groups"] == count["unchanged_groups"]
        assert expected["permutation_counts"] == [count["permutations"][i] for i in range(len(CATALOG[len(MODES[mode])]))]
        summary[mode] = {**count, "input_stream_sha256": streams[mode].hexdigest()}
    return summary, sample_oracle, permitted


def verify_derived(normal_source, derived, diff):
    from difflib import unified_diff

    orig = normal_source.read_text()
    assert sha_file(normal_source) == ORIGINAL_ENTRY
    actual = derived.read_text()
    old_tree, new_tree = (ast.parse(file_text) for file_text in (orig, actual))
    for name in ("strict_reload", "catalog_and_users", "normalized_prediction", "audit_prediction", "collate"):
        definitions = [[node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
                       for tree in (old_tree, new_tree)]
        assert len(definitions[0]) == len(definitions[1]) == 1
        assert ast.dump(definitions[0][0]) == ast.dump(definitions[1][0]), name
    calls = [
        [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and isinstance(n.func.value, ast.Name) and n.func.value.id == "model" and n.func.attr == "generate"]
        for tree in (old_tree, new_tree)
    ]
    assert len(calls[0]) == len(calls[1]) == 1 and ast.dump(calls[0][0]) == ast.dump(calls[1][0])
    saved = diff.read_text().splitlines(keepends=True)
    expected = list(unified_diff(orig.splitlines(keepends=True), actual.splitlines(keepends=True)))
    assert saved[2:] == expected[2:]


def independent_gpu(gpu, normal, sample_oracle, permitted, normal_source):
    result = {}
    normal_rank = read(normal / "rank0.json")
    matching_rank_fields = ("entry_sha256", "allowed_callback_sha256", "adapter_files", "data_identity",
                            "author_sources", "model", "strict_adapter_reload", "tokenizer", "versions")
    run_versions, derived_digests, parameter_digests = set(), set(), set()
    for mode in MODES:
        shard = gpu / mode / "predictions.rank0.jsonl"
        if not shard.is_file():
            shard = gpu / mode / "test" / "predictions.rank0.jsonl"
        rank = read(shard.with_name("rank0.json"))
        assert rank["status"] == "PREFLIGHT_INTERFACE_PASS" and rank["preflight_only"]
        assert rank["rank"] == rank["local_rank"] == 0 and rank["world_size"] == 1
        assert rank["preflight_assignment"] == HISTORICAL_ROWS and rank["completed_users"] == rank["assigned_users"] == 32
        assert rank["gpu"]["visible_devices"].startswith("GPU-")
        for key in matching_rank_fields:
            assert rank[key] == normal_rank[key], key
        run_versions.add(rank["RUN_COMMIT"])
        derived = shard.with_name("derived.rank0.py")
        if normal_source:
            verify_derived(normal_source, derived, shard.with_name("derived.rank0.diff"))
        derived_digests.add(sha_file(derived))
        shuffle = rank["shuffle"]
        assert shuffle["actual_model_module"] == "Qwen3Model" and shuffle["sdpa"] and not shuffle["quantized"]
        assert shuffle["generate_calls"] == 32 and shuffle["parameter_versions_unchanged"]
        assert shuffle["parameter_sha256_before"] == shuffle["parameter_sha256_after"]
        assert len(shuffle["parameter_sha256_before"]) == 64
        parameter_digests.add(shuffle["parameter_sha256_before"])
        expected_overrides = {
            "num_beams": 20, "num_return_sequences": 20, "max_new_tokens": 10, "early_stopping": True,
            "do_sample": False, "use_cache": True, "output_scores": True, "return_dict_in_generate": True,
            "eos_token_id": EOS, "pad_token_id": EOS,
        }
        assert shuffle["generation_scalar_overrides"] == expected_overrides
        assert all(shuffle["effective_generation_config"][key] == v for key, v in expected_overrides.items())
        assert shuffle["effective_generation_config"]["cache_implementation"] is None
        modes = list(rows(shard))
        assert [r["row_idx"] for r in modes] == HISTORICAL_ROWS
        natural_identity = []
        for row in modes:
            i = row["row_idx"]
            frozen = sample_oracle[i]["normal"]
            transformed, groups = sample_oracle[i]["modes"][mode]
            assert row["user_id"] == frozen["user_id"] and row["target_item_id"] == frozen["target_item_id"]
            assert row["target"] == frozen["target"] and row["rank"] == 0 and row["prompt_id"] == 0
            assert row["input_token_ids"] == transformed
            assert row["input_nonpadding_tokens"] == len(transformed) and row["original_callback_verified_all_tokens"]
            verify_candidates(row, permitted)
            evidence = row["shuffle"]
            assert evidence["input_changed"] == (transformed != frozen["input_token_ids"])
            assert evidence["original_input_ids"] == frozen["input_token_ids"]
            assert evidence["history_group_count"] == len(groups)
            assert evidence["protocol"] == PROTOCOL and evidence["identity_policy"] == "allow_identity"
            assert evidence["mode"] == mode and evidence["seed"] == 42
            for actual, expected in zip(evidence["groups"], groups, strict=True):
                assert all(actual[key] == value for key, value in expected.items())
            prefill = evidence["actual_prefill"]
            assert evidence["eval_mode"] and not evidence["grad_enabled"]
            assert evidence["sequence_score_dtype"] == "torch.float32"
            callback = evidence["callback"]
            assert callback["original_validation"] and callback["bound_prefix_sha256"] == digest(transformed)
            assert callback["allowed_counts"] == [150, 256, 256, 256, 1]
            assert callback["separator_ids"] and transformed[-len(callback["separator_ids"]):] == callback["separator_ids"]
            if "separator_ids" in sample_oracle[i]:
                assert callback["separator_ids"] == sample_oracle[i]["separator_ids"]
            assert len(callback["generation_position_calls"]) == 5 and all(n > 0 for n in callback["generation_position_calls"])
            assert prefill["all_beams_ids_mask_positions_exact"]
            assert prefill["lookup_exact"] is True and prefill["lookup_shape"] == [20, len(transformed), 1024]
            assert prefill["lookup_dtype"] == "torch.bfloat16"
            assert prefill["lookup_sha256"] == prefill["expected_lookup_sha256"] and len(prefill["lookup_sha256"]) == 64
            assert prefill["cache_length"] == 0 and prefill["beam_expansion"] == 20
            assert prefill["input_ids"] == transformed and prefill["input_sha256"] == digest(transformed)
            explicit_mask = prefill["attention_mask"]
            assert explicit_mask == [1] * len(transformed)
            assert prefill["attention_mask_sha256"] == digest(explicit_mask)
            positions = list(range(len(transformed)))
            assert prefill["position_ids"] == positions and prefill["position_ids_sha256"] == digest(positions)
            assert prefill["cache_position"] == positions
            assert prefill["initial_cache_type"] == "DynamicCache"
            cache_layer_lengths = prefill["initial_cache_layer_lengths"]
            assert cache_layer_lengths == [0] * 28
            if evidence["input_changed"]:
                continue
            diffs = Counter()
            for candidate, reference in zip(row["candidates"], frozen["candidates"], strict=True):
                for field in ("generated_token_ids", "sid", "generation_order", "raw_decoded", "sequence_score", "eos_position"):
                    diffs[field] += candidate[field] != reference[field]
            max_abs = max(abs(c["sequence_score"] - b["sequence_score"])
                          for c, b in zip(row["candidates"], frozen["candidates"], strict=True))
            natural_identity.append({"row_idx": i, "candidate_field_differences": diffs,
                                     "sorted_order_matches": row["sorted_generation_order"] == frozen["sorted_generation_order"],
                                     "score_max_abs_difference": max_abs})
        result[mode] = {"rows": 32, "candidates": 640, "natural_identity": natural_identity,
                        "raw_sha256": sha_file(shard)}
    assert len(derived_digests) == len(run_versions) == len(parameter_digests) == 1
    return {"results": result, "runtime_commit": next(iter(run_versions)),
            "unique_derived_sha256": next(iter(derived_digests)),
            "unique_across_modes_parameter_sha256": next(iter(parameter_digests))}


def main():
    parser = ArgumentParser(description=__doc__)
    for arg in ("data", "normal", "cpu", "gpu", "tokenizer", "normal-source"):
        parser.add_argument("--" + arg, type=Path, required=arg in {"data", "normal", "cpu", "tokenizer"})
    args = parser.parse_args()
    if args.gpu and not args.normal_source:
        parser.error("--gpu requires --normal-source")
    cpu, frozen, permitted = independent_cpu(args.data, args.normal, args.cpu, args.tokenizer)
    out = {"cpu_modes": cpu, "user_modes": 99088, "history_group_modes": 691716,
           "independent_prompt0_tokenizer": args.tokenizer is not None}
    if args.gpu:
        out["gpu"] = independent_gpu(args.gpu, args.normal, frozen, permitted, args.normal_source)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
