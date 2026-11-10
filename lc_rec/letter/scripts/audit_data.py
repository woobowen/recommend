#!/usr/bin/env python3
"""CPU-only LETTER SeqRec asset, tokenizer, split, mask and length audit.

No model weights are loaded, no metric is computed, and no source asset is edited.
Lengths for prompt 0 are exhaustively tokenized. Other prompts use finite
history-length classes only after checking the added-token substitution invariant;
the report distinguishes direct measurements from those derived lengths.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import resource
import socket
import sys
import time
from types import SimpleNamespace


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def data_directory(root, dataset):
    root = Path(root).resolve()
    if (root / (dataset + ".inter.json")).is_file():
        return root
    if (root / dataset / (dataset + ".inter.json")).is_file():
        return root / dataset
    raise FileNotFoundError(f"No {dataset}.inter.json directly in {root} or {root / dataset}")


def load_assets(root, dataset):
    directory = data_directory(root, dataset)
    data, identity = {}, {}
    for kind in ("inter", "item", "index"):
        path = directory / f"{dataset}.{kind}.json"
        raw = path.read_bytes()
        if raw.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"LFS pointer instead of asset: {path}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"Expected mapping in {path}")
        data[kind] = value
        identity[kind] = {"path": str(path), "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "canonical_sha256": canonical_sha(value), "entries": len(value)}
    data["directory"] = directory
    return data, identity


def axis_summary(values):
    values = set(values)
    return {"count": len(values), "sorted_sha256": canonical_sha(sorted(values)),
            "first_10": sorted(values)[:10]}


def mismatch_summary(left, right):
    common = set(left) & set(right)
    differing = sorted(key for key in common if left[key] != right[key])
    return {"equal": left == right, "same_key_set": set(left) == set(right),
            "same_key_order": list(left) == list(right), "common_count": len(common),
            "differing_common_values": len(differing), "differing_examples": differing[:20],
            "only_a": axis_summary(set(left) - set(right)),
            "only_b": axis_summary(set(right) - set(left))}


def asset_summary(asset):
    inters, items, indices = asset["inter"], asset["item"], asset["index"]
    used = {str(i) for sequence in inters.values() for i in sequence}
    sid_groups, joined_groups, position_tokens = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(set)
    for item, sid in indices.items():
        if not isinstance(sid, list) or not sid or not all(isinstance(t, str) and t for t in sid):
            raise ValueError(f"Malformed SID for item {item}")
        sid_groups[tuple(sid)].append(item)
        joined_groups["".join(sid)].append(item)
        for position, token in enumerate(sid):
            position_tokens[position].add(token)
    collisions = [group for group in sid_groups.values() if len(group) > 1]
    joined_collisions = [group for group in joined_groups.values() if len(group) > 1]
    lengths = [len(s) for s in inters.values()]
    if not lengths or min(lengths) < 3:
        raise ValueError("SeqRec leave-two-out requires every sequence to contain at least 3 items")
    missing = used - set(indices)
    if missing:
        raise ValueError(f"Interactions refer to {len(missing)} missing SID keys: {sorted(missing)[:20]}")
    return {"users": axis_summary(inters), "items": axis_summary(items), "index_items": axis_summary(indices),
            "interaction_items": axis_summary(used), "interactions": sum(lengths),
            "min_sequence_length": min(lengths), "max_sequence_length": max(lengths),
            "sequence_length_histogram": dict(sorted(collections.Counter(lengths).items())),
            "train_rows": sum(max(n - 3, 0) for n in lengths), "valid_rows": len(inters), "test_rows": len(inters),
            "sid_length_histogram": dict(sorted(collections.Counter(map(len, indices.values())).items())),
            "sid_utf8_byte_length_histogram": dict(sorted(collections.Counter(len("".join(v).encode()) for v in indices.values()).items())),
            "unique_sid_tokens": len({t for sid in indices.values() for t in sid}),
            "position_token_counts": {str(p): len(tokens) for p, tokens in position_tokens.items()},
            "tuple_collision_groups": len(collisions), "tuple_colliding_items": sum(map(len, collisions)),
            "tuple_collision_examples": collisions[:20], "joined_collision_groups": len(joined_collisions),
            "joined_collision_examples": joined_collisions[:20],
            "index_without_metadata": axis_summary(set(indices) - set(items)),
            "metadata_without_index": axis_summary(set(items) - set(indices)),
            "used_without_metadata": axis_summary(used - set(items))}


def split_records(asset, maximum_history):
    result = {"train": [], "valid": [], "test": []}
    for user_ordinal, (uid, items) in enumerate(asset["inter"].items()):
        for mode, positions in (("train", range(1, len(items) - 2)),
                                ("valid", [len(items) - 2]), ("test", [len(items) - 1])):
            for position in positions:
                begin = max(0, position - maximum_history) if maximum_history > 0 else 0
                history = [str(item) for item in items[begin:position]]
                target = str(items[position])
                result[mode].append({"row_idx": len(result[mode]), "user_ordinal": user_ordinal,
                                     "user_id": uid, "target_position": position,
                                     "history_item_ids": history, "target_item_id": target,
                                     "history_len": len(history)})
    return result


def compare_split_labels(a, b):
    result = {}
    for mode in ("train", "valid", "test"):
        a_map = {(r["user_id"], r["target_position"]): (r["history_item_ids"], r["target_item_id"]) for r in a[mode]}
        b_map = {(r["user_id"], r["target_position"]): (r["history_item_ids"], r["target_item_id"]) for r in b[mode]}
        common = a_map.keys() & b_map.keys()
        result[mode] = {"a_count": len(a_map), "b_count": len(b_map),
                        "same_label_and_history_by_user_position": a_map == b_map,
                        "differing_common_rows": sum(a_map[key] != b_map[key] for key in common),
                        "common_rows": len(common), "only_a_rows": len(a_map.keys() - b_map.keys()),
                        "only_b_rows": len(b_map.keys() - a_map.keys()),
                        "a_labels_sha256": canonical_sha([[r["user_id"], r["target_position"], r["target_item_id"]] for r in a[mode]]),
                        "b_labels_sha256": canonical_sha([[r["user_id"], r["target_position"], r["target_item_id"]] for r in b[mode]])}
    return result


def split_sid_summary(asset, splits):
    sid_strings = {item: "".join(sid) for item, sid in asset["index"].items()}
    multiplicity = collections.Counter(sid_strings.values())
    result = {}
    for mode, rows in splits.items():
        targets = [sid_strings[r["target_item_id"]] for r in rows]
        histories = [[sid_strings[item] for item in r["history_item_ids"]] for r in rows]
        result[mode] = {"target_sid_sha256": canonical_sha(targets),
                        "history_sid_sha256": canonical_sha(histories),
                        "target_rows_with_nonunique_sid": sum(multiplicity[sid] > 1 for sid in targets),
                        "all_target_items_mapped": True, "all_history_items_mapped": True}
    return result


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(values, np):
    flat = np.asarray(values).reshape(-1)
    return {"count": int(flat.size), "min": int(flat.min()), "max": int(flat.max()),
            "mean": float(flat.mean()), "std": float(flat.std()),
            "quantiles": {str(q): float(np.quantile(flat, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "histogram": {str(int(v)): int(n) for v, n in zip(*np.unique(flat, return_counts=True))}}


def make_text(record, prompt, template):
    instruction = prompt["instruction"].format(**record)
    response = prompt["response"].format(**record)
    input_text = template.format(instruction=instruction, response="")
    full_text = template.format(instruction=instruction, response=response)
    return {"input_ids": input_text, "labels": full_text}


def check_budget(start, seconds, minimum_gib):
    if time.monotonic() - start > seconds:
        raise TimeoutError(f"CPU audit exceeded --max-seconds={seconds}")
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            if available < minimum_gib * 1024**3:
                raise MemoryError(f"MemAvailable={available / 1024**3:.2f}GiB below {minimum_gib}GiB floor")
            return


def measure_pairs(tokenizer, pairs, batch_size, guard, np):
    input_lengths, full_lengths = [], []
    for begin in range(0, len(pairs), batch_size):
        guard()
        batch = pairs[begin:begin + batch_size]
        inputs = tokenizer([p["input_ids"] for p in batch], padding=False, truncation=False)["input_ids"]
        full = tokenizer([p["labels"] + tokenizer.eos_token for p in batch], padding=False, truncation=False)["input_ids"]
        input_lengths.extend(map(len, inputs))
        full_lengths.extend(map(len, full))
        if begin % (batch_size * 16) == 0:
            print(json.dumps({"event": "length_batch", "done": min(begin + len(batch), len(pairs)), "total": len(pairs)}), flush=True)
    return np.asarray(input_lengths, dtype=np.int32), np.asarray(full_lengths, dtype=np.int32)


def audit_tokenizer(tokenizer, indices):
    tokens = sorted({token for sid in indices.values() for token in sid})
    base_size = len(tokenizer)
    preexisting = [token for token in tokens if token in tokenizer.get_vocab()]
    added = tokenizer.add_tokens(tokens)
    failures = []
    manifest = {}
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id] or token_id == tokenizer.unk_token_id:
            failures.append({"token": token, "id": token_id, "encoded": encoded})
        manifest[token] = int(token_id)
    sid_ids = {}
    for item, sid in indices.items():
        expected = [manifest[t] for t in sid]
        actual = tokenizer.encode("".join(sid), add_special_tokens=False)
        if actual != expected:
            failures.append({"item": item, "expected": expected, "encoded": actual})
        sid_ids[item] = expected
    if failures:
        raise ValueError(f"SID tokenization failures: {failures[:10]}")
    return {"base_length": base_size, "new_tokens_count": added, "expanded_length": len(tokenizer),
            "new_token_order": "Python sorted(set(all SID token strings)); lexical, not numerical",
            "preexisting_sid_token_count": len(preexisting), "preexisting_sid_token_examples": preexisting[:10],
            "atomic_token_checks": len(tokens), "concatenated_sid_checks": len(indices),
            "atomic_and_concatenated_sid_checks_passed": True,
            "token_to_id": manifest, "sid_token_id_map_sha256": canonical_sha(sid_ids),
            "special_tokens": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                               "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id}}, sid_ids


def original_dataset_checks(source_module, a, splits, dataset_args, np, seed):
    datasets = {}
    for mode in ("train", "valid", "test"):
        if mode == "valid":
            np.random.seed(seed)
        datasets[mode] = source_module.SeqRecDataset(dataset_args, mode=mode,
                                                    prompt_sample_num=2 if mode == "valid" else 1)
        original = datasets[mode].inter_data
        if len(original) != len(splits[mode]):
            raise AssertionError(f"Original {mode} row count mismatch")
        for actual, record in zip(original, splits[mode]):
            expected = {"item": "".join(a["index"][record["target_item_id"]]),
                        "inters": dataset_args.his_sep.join("".join(a["index"][item]) for item in record["history_item_ids"])}
            if dataset_args.add_prefix:
                expected["inters"] = dataset_args.his_sep.join(f"{j + 1}. " + "".join(a["index"][item]) for j, item in enumerate(record["history_item_ids"]))
            if actual != expected:
                raise AssertionError(f"Original {mode} row {record['row_idx']} differs from explicit leave-two-out reconstruction")
    tokens = sorted({t for sid in a["index"].values() for t in sid})
    if datasets["train"].get_new_tokens() != tokens:
        raise AssertionError("get_new_tokens is not the audited sorted token sequence")
    return datasets


def finite_length_tables(tokenizer, prompts, template, datasets, splits, args, guard, np):
    """Map every real row to a finite history-length class, then verify on text.

    All SID tokens must be newly-added atomic tokens with uniform SID length.
    Splitting text on those tokens leaves identical non-SID fragments for a fixed
    history length and prompt. The tokenizer's added-token trie therefore sees
    exactly the same non-SID segments. Prompt 0 is checked on EVERY real row;
    each remaining prompt is checked on deterministic real rows of EVERY class.
    The finite maxima are a conditional exhaustive bound, not sampled maxima.
    """
    train = datasets["train"]
    sid_sizes = {len(re.findall(r"<[^<>]+>", row["item"])) for row in train.inter_data}
    if len(sid_sizes) != 1:
        raise ValueError("Finite-class length mode requires uniform SID length")
    sid_size = next(iter(sid_sizes))
    sid_tokens = train.get_new_tokens()
    if any(not re.fullmatch(r"<[^<>]+>", t) for t in sid_tokens):
        raise ValueError("Finite-class mode requires <...> atomic SID tokens")
    if any(t not in tokenizer.get_added_vocab() for t in sid_tokens):
        raise ValueError("Finite-class mode requires every SID token in tokenizer added vocabulary")
    canonical_sid = sid_tokens[0] * sid_size
    normalize = lambda text: re.sub(r"<[^<>]+>", sid_tokens[0], text)
    all_history_lengths = sorted({r["history_len"] for records in splits.values() for r in records})
    canonical = {}
    for history_len in all_history_lengths:
        history = [canonical_sid] * history_len
        if args.add_prefix:
            history = [f"{i + 1}. {item}" for i, item in enumerate(history)]
        canonical[history_len] = {"item": canonical_sid, "inters": args.his_sep.join(history)}
    tables_input, tables_full = {}, {}
    for history_len, row in canonical.items():
        pairs = [make_text(row, p, template) for p in prompts]
        inputs, full = measure_pairs(tokenizer, pairs, args.batch_size, guard, np)
        tables_input[history_len], tables_full[history_len] = inputs, full
    # Prove that all source-produced rows reduce to one of the measured classes.
    normalized_rows = 0
    for mode, dataset in datasets.items():
        for row, record in zip(dataset.inter_data, splits[mode]):
            expected = canonical[record["history_len"]]
            if {key: normalize(value) for key, value in row.items()} != expected:
                raise AssertionError(f"Finite-class non-SID segment invariant fails in {mode} row {record['row_idx']}")
            normalized_rows += 1
    matrices = {}
    direct = {}
    for mode in ("train", "valid", "test"):
        guard()
        input_matrix = np.stack([tables_input[r["history_len"]] for r in splits[mode]])
        full_matrix = np.stack([tables_full[r["history_len"]] for r in splits[mode]])
        pairs = [make_text(row, prompts[0], template) for row in datasets[mode].inter_data]
        print(json.dumps({"event": "exhaustive_prompt0", "split": mode, "rows": len(pairs)}), flush=True)
        measured_inputs, measured_full = measure_pairs(tokenizer, pairs, args.batch_size, guard, np)
        if not np.array_equal(measured_inputs, input_matrix[:, 0]) or not np.array_equal(measured_full, full_matrix[:, 0]):
            raise AssertionError(f"Finite-class lengths disagree with exhaustive {mode} prompt0 tokenization")
        matrices[mode] = {"input": input_matrix, "full": full_matrix}
        direct[mode] = {"prompt0_rows": len(pairs), "input_and_full_tokenized": True}
    # For every prompt/history class, check multiple actual SID identities.
    rng = np.random.RandomState(args.seed)
    verification_pairs, expected_input, expected_full = [], [], []
    for mode in ("train", "valid", "test"):
        groups = collections.defaultdict(list)
        for row in splits[mode]:
            groups[row["history_len"]].append(row["row_idx"])
        for history_len, rows in sorted(groups.items()):
            selected = sorted(set([rows[0], rows[-1]]) | set(rng.choice(rows, min(16, len(rows)), replace=False).tolist()))
            for row_idx in selected:
                for prompt_id in range(1, len(prompts)):
                    verification_pairs.append(make_text(datasets[mode].inter_data[row_idx], prompts[prompt_id], template))
                    expected_input.append(tables_input[history_len][prompt_id])
                    expected_full.append(tables_full[history_len][prompt_id])
    vi, vf = measure_pairs(tokenizer, verification_pairs, args.batch_size, guard, np)
    if not np.array_equal(vi, expected_input) or not np.array_equal(vf, expected_full):
        raise AssertionError("Finite-class lengths disagree with measured nonzero prompt examples")
    return matrices, {"method": "exhaustive prompt0 tokenization plus verified finite history-length classes",
                      "direct_measurements": direct, "other_prompt_real_text_pairs_checked": len(verification_pairs),
                      "all_source_rows_normalized_and_checked": normalized_rows,
                      "classes": all_history_lengths, "uniform_sid_tokens_per_item": sid_size,
                      "per_class_input_lengths": {str(h): v.tolist() for h, v in tables_input.items()},
                      "per_class_full_lengths_including_eos": {str(h): v.tolist() for h, v in tables_full.items()},
                      "conditional_exhaustive_max_full_length": int(max(v.max() for v in tables_full.values())),
                      "bound_conditions": ["all SID pieces are newly-added atomic tokens", "uniform SID length",
                                           "each real row reduces to the measured non-SID string for its history length",
                                           "fixed prompt and separator text", "original LlamaTokenizer added-token splitting"],
                      "not_performed": "independent full-tokenization of every row under every nonzero prompt"}


def choose_samples(split, matrix, np):
    selections = {}
    flattened = matrix["full"].reshape(-1)
    prompt_count = matrix["full"].shape[1]
    for name, quantile in (("q10", .1), ("q50", .5), ("q90", .9), ("max", 1.0)):
        nearest_index = int(round(quantile * (flattened.size - 1)))
        length = int(np.partition(flattened, nearest_index)[nearest_index])
        flat_index = int(np.flatnonzero(flattened == length)[0])
        row_idx, prompt_id = divmod(flat_index, prompt_count)
        selections[name] = dict(split[row_idx], prompt_id=prompt_id,
                                input_length=int(matrix["input"][row_idx, prompt_id]),
                                full_length_including_eos=int(matrix["full"][row_idx, prompt_id]))
    return selections


def audit_masks(collator_module, tokenizer, train, selections, prompts, template, args, torch):
    pairs = [make_text(train.inter_data[s["row_idx"]], prompts[s["prompt_id"]], template) for s in selections.values()]
    args.only_train_response = True
    original_collator = collator_module.Collator(args, tokenizer)
    batch = original_collator(pairs)
    checks = []
    for index, (name, selected) in enumerate(selections.items()):
        pair = pairs[index]
        input_ids = tokenizer(pair["input_ids"])["input_ids"]
        full_ids = tokenizer(pair["labels"] + tokenizer.eos_token)["input_ids"]
        if full_ids[:len(input_ids)] != input_ids:
            raise AssertionError(f"Input tokenization is not a full text token prefix for {name}")
        actual = batch["labels"][index].tolist()
        expected = [-100] * len(input_ids) + full_ids[len(input_ids):]
        expected += [-100] * (len(actual) - len(expected))
        if actual != expected:
            raise AssertionError(f"Response-only mask mismatch in {name}: {actual} != {expected}")
        supervised = [i for i in actual if i != -100]
        sid_target = train.inter_data[selected["row_idx"]]["item"]
        expected_sid = tokenizer.encode(sid_target, add_special_tokens=False)
        if supervised != expected_sid + [tokenizer.eos_token_id]:
            raise AssertionError(f"Unexpected response targets in {name}: {supervised}")
        if not torch.equal(batch["attention_mask"][index].sum(), torch.tensor(len(full_ids))):
            raise AssertionError("Attention mask length mismatch")
        checks.append({"selection": name, "row_idx": selected["row_idx"], "prompt_id": selected["prompt_id"],
                       "input_length": len(input_ids), "full_length": len(full_ids),
                       "supervised_count": len(supervised), "supervised_ids": supervised,
                       "decoded_supervised": tokenizer.decode(supervised), "mask_exact_match": True})
    return {"original_collator_executed": True, "batch_shape": list(batch["input_ids"].shape), "checks": checks,
            "padding_side": tokenizer.padding_side, "only_train_response": True,
            "source_behavior": "copy full input_ids; mask all pad IDs and coordinates occupied by input-only target tokenization"}


def audit_evaluation_mapping(datasets, tokenizer, sid_ids, a, prompts, template, torch):
    dataset = datasets["test"]
    all_items = dataset.get_all_items()
    expected_items = {"".join(sid) for sid in a["index"].values()}
    if all_items != expected_items:
        raise AssertionError("Evaluator candidate set differs from training index")
    allowed_fn = dataset.get_prefix_allowed_tokens_fn(tokenizer)
    expected_bins = collections.defaultdict(set)
    for ids in sid_ids.values():
        for position, token in enumerate(ids):
            expected_bins[position].add(token)
    length_set = {len(v) for v in sid_ids.values()}
    if len(length_set) != 1:
        raise AssertionError("Original position constraint assumes one SID length")
    sid_length = next(iter(length_set))
    expected_bins[sid_length] = {tokenizer.eos_token_id}
    if dataset.allowed_tokens != expected_bins:
        raise AssertionError("Actual generation position mask differs from audited SID token IDs")
    checked = 0
    first_row = dataset.inter_data[0]
    target_ids = tokenizer.encode(first_row["item"], add_special_tokens=False)
    for prompt in prompts:
        input_ids = tokenizer(make_text(first_row, prompt, template)["input_ids"])["input_ids"]
        for position in range(sid_length + 1):
            actual = set(allowed_fn(0, torch.tensor(input_ids + target_ids[:position])))
            if actual != expected_bins[position]:
                raise AssertionError("Response-prefix generation constraint mismatch")
            checked += 1
    decode_failures = []
    for item, ids in sid_ids.items():
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip().replace(" ", "")
        if decoded != "".join(a["index"][item]):
            decode_failures.append(item)
    if decode_failures:
        raise AssertionError(f"Evaluator SID roundtrip failure: {decode_failures[:10]}")
    cross_product = 1
    for position in range(sid_length):
        cross_product *= len(expected_bins[position])
    return {"same_index_for_train_input_response_test_target_candidates_and_constraint": True,
            "all_sid_decode_normalize_roundtrips": len(sid_ids), "response_prefix_cases_checked": checked,
            "position_mask_sizes": {str(p): len(ids) for p, ids in expected_bins.items()},
            "constraint_type": "position-wise vocabulary sets, not an item-prefix trie",
            "position_mask_cross_product": cross_product, "actual_unique_sid_candidates": len(all_items),
            "invalid_combinations_are_possible": cross_product > len(all_items),
            "generation_source_max_new_tokens": 10, "generation_source_num_beams_default": 20,
            "test_sampler_source_drop_last": True,
            "two_rank_original_sampler_test_users_per_prompt": (len(dataset) // 2) * 2,
            "two_rank_dropped_user_count_per_prompt": len(dataset) % 2,
            "metric_computation_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Immutable original LETTER-LC-Rec source directory")
    parser.add_argument("--a-data", required=True, help="LETTER data parent or Instruments directory")
    parser.add_argument("--b-data", required=True, help="Original LC-Rec data parent or Instruments directory")
    parser.add_argument("--model", required=True, help="Existing fixed LLaMA model directory; tokenizer files only")
    parser.add_argument("--output", required=True, help="New isolated audit output directory")
    parser.add_argument("--dataset", default="Instruments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-his-len", type=int, default=20)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--his-sep", default=", ")
    parser.add_argument("--add-prefix", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--min-available-gib", type=float, default=40.0)
    args = parser.parse_args()
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "letter_seqrec_data_audit_v1", "status": "RUNNING",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": sys.argv,
              "interpreter": sys.executable, "cwd": os.getcwd(), "host": socket.gethostname(),
              "script_path": str(Path(__file__).resolve()), "script_sha256": sha_file(__file__),
              "effective_args": vars(args), "model_weights_loaded": False, "gpu_execution": False}
    write_json(output / "audit.json", report)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        guard = lambda: check_budget(started, args.max_seconds, args.min_available_gib)
        guard()
        source = Path(args.source).resolve()
        report["original_source"] = {name: {"path": str(source / name), "sha256": sha_file(source / name)}
                                     for name in ("data.py", "prompt.py", "collator.py", "utils.py", "lora_finetune.py",
                                                  "test_ddp.py", "evaluate.py", "modeling_letter.py", "run_train.sh", "run_test_ddp.sh")}
        a, a_identity = load_assets(args.a_data, args.dataset)
        b, b_identity = load_assets(args.b_data, args.dataset)
        a_splits = split_records(a, args.max_his_len)
        b_splits = split_records(b, args.max_his_len)
        report["assets"] = {"a": {"identity": a_identity, "summary": asset_summary(a)},
                            "b": {"identity": b_identity, "summary": asset_summary(b)}}
        report["comparison"] = {kind: mismatch_summary(a[kind], b[kind]) for kind in ("inter", "item", "index")}
        common = set(a["index"]) & set(b["index"])
        report["comparison"]["sid_same_item_values"] = {"same": sum(a["index"][i] == b["index"][i] for i in common),
                                                        "different": sum(a["index"][i] != b["index"][i] for i in common),
                                                        "common_items": len(common)}
        report["comparison"]["split_labels"] = compare_split_labels(a_splits, b_splits)
        report["split_sid_identity"] = {"a": split_sid_summary(a, a_splits), "b": split_sid_summary(b, b_splits)}
        report["comparison"]["mapping_identity_limit"] = "Exact per-item metadata and per-user ordered interaction comparison establishes only the supplied public ID axes; no unavailable historical ASIN/user remapping or checkpoint identity is inferred."
        write_json(output / "audit.json", report)
        guard()
        import numpy as np
        import torch
        from transformers import LlamaTokenizer
        report["versions"] = {}
        for package in ("transformers", "tokenizers", "sentencepiece", "torch", "numpy"):
            report["versions"][package] = importlib.metadata.version(package)
        sys.path.insert(0, str(source))
        original_data = import_source("_letter_original_data_audit", source / "data.py")
        original_collator = import_source("_letter_original_collator_audit", source / "collator.py")
        prompt_module = import_source("_letter_original_prompt_audit", source / "prompt.py")
        prompts, template = prompt_module.all_prompt["seqrec"], prompt_module.sft_prompt
        report["seqrec_prompt_count"] = len(prompts)
        dataset_args = SimpleNamespace(dataset=args.dataset, data_path=str(a["directory"].parent),
                                       max_his_len=args.max_his_len, his_sep=args.his_sep,
                                       index_file=".index.json", add_prefix=args.add_prefix,
                                       sample_valid=True, valid_prompt_id=0,
                                       only_train_response=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        datasets = original_dataset_checks(original_data, a, a_splits, dataset_args, np, args.seed)
        report["original_dataset_execution"] = {"all_rows_match_explicit_split_reconstruction": True,
                                                "train_rows": len(datasets["train"]),
                                                "valid_rows_after_two_prompts": len(datasets["valid"]),
                                                "test_rows": len(datasets["test"]),
                                                "auxiliary_user_intention_files_opened": False,
                                                "training_prompt_behavior": "Python random.randint over 17 templates at each __getitem__; sample_num is unused by SeqRec training",
                                                "valid_prompt_behavior": "NumPy choice without replacement: two prompts per user, constructed once after np.random.seed(42)"}
        model = Path(args.model).resolve()
        report["base_tokenizer_assets"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha_file(p)}
                                           for p in model.iterdir() if p.is_file() and p.name in
                                           ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")}
        tokenizers, sid_id_maps = {}, {}
        for name, asset in (("a", a), ("b", b)):
            guard()
            tokenizer = LlamaTokenizer.from_pretrained(str(model), model_max_length=args.model_max_length,
                                                       padding_side="right", local_files_only=True)
            tokenizer.pad_token_id = 0
            info, sid_ids = audit_tokenizer(tokenizer, asset["index"])
            tokenizers[name], sid_id_maps[name] = tokenizer, sid_ids
            write_json(output / f"tokenizer_{name}_ids.json", info)
            report[f"tokenizer_{name}"] = {key: value for key, value in info.items() if key != "token_to_id"}
        tokenizer = tokenizers["a"]
        tokenizer.save_pretrained(output / "tokenizer_a")
        reloaded = LlamaTokenizer.from_pretrained(output / "tokenizer_a", local_files_only=True)
        if reloaded.get_vocab() != tokenizer.get_vocab() or reloaded.pad_token_id != tokenizer.pad_token_id:
            raise AssertionError("Saved/reloaded expanded tokenizer differs")
        report["tokenizer_a_save_reload"] = {"path": str(output / "tokenizer_a"), "exact_vocab_and_pad_id_match": True}
        report["tokenizer_implementation"] = {}
        for label, function in (("tokenize", tokenizer.tokenize), ("_tokenize", tokenizer._tokenize)):
            code = inspect.getsource(function)
            path = output / ("tokenizer_implementation_" + label + ".py.txt")
            path.write_text(code)
            report["tokenizer_implementation"][label] = {"path": str(path), "sha256": sha_file(path)}
        write_json(output / "audit.json", report)
        matrices, length_method = finite_length_tables(tokenizer, prompts, template, datasets, a_splits, args, guard, np)
        report["length_measurement"] = length_method
        report["length_distributions"] = {}
        for mode, matrix in matrices.items():
            report["length_distributions"][mode] = {"input_per_prompt": [distribution(matrix["input"][:, i], np) for i in range(len(prompts))],
                                                    "full_with_eos_per_prompt": [distribution(matrix["full"][:, i], np) for i in range(len(prompts))],
                                                    "all_prompt_input": distribution(matrix["input"], np),
                                                    "all_prompt_full_with_eos": distribution(matrix["full"], np)}
        # Recover exact prompt IDs chosen by original validation construction.
        rng = np.random.RandomState(args.seed)
        valid_prompt_ids = np.stack([rng.choice(len(prompts), 2, replace=False) for _ in a_splits["valid"]])
        valid_pairs = datasets["valid"].valid_text_data
        for row_idx, prompt_ids in enumerate(valid_prompt_ids):
            for j, prompt_id in enumerate(prompt_ids):
                if valid_pairs[2 * row_idx + j] != make_text(datasets["valid"].inter_data[row_idx], prompts[prompt_id], template):
                    raise AssertionError("Validation prompt RNG binding mismatch")
        valid_input = np.take_along_axis(matrices["valid"]["input"], valid_prompt_ids, axis=1)
        valid_full = np.take_along_axis(matrices["valid"]["full"], valid_prompt_ids, axis=1)
        report["valid_seed42_two_prompt_distribution"] = {"input": distribution(valid_input, np),
                                                         "full_with_eos": distribution(valid_full, np),
                                                         "prompt_id_histogram": dict(sorted(collections.Counter(map(int, valid_prompt_ids.reshape(-1))).items())),
                                                         "exact_original_valid_text_rng_match": True}
        selections = choose_samples(a_splits["train"], matrices["train"], np)
        report["mask_audit"] = audit_masks(original_collator, tokenizer, datasets["train"], selections,
                                           prompts, template, dataset_args, torch)
        report["evaluation_mapping"] = audit_evaluation_mapping(datasets, tokenizer, sid_id_maps["a"], a, prompts, template, torch)
        rng = np.random.RandomState(args.seed)
        mixed_rows = rng.choice(len(a_splits["train"]), min(128, len(a_splits["train"])), replace=False)
        mixed_prompts = rng.randint(len(prompts), size=len(mixed_rows))
        mixed = [dict(a_splits["train"][int(row_idx)], prompt_id=int(prompt_id),
                      input_length=int(matrices["train"]["input"][row_idx, prompt_id]),
                      full_length_including_eos=int(matrices["train"]["full"][row_idx, prompt_id]))
                 for row_idx, prompt_id in zip(mixed_rows, mixed_prompts)]
        probe = {"schema": "letter_seqrec_probe_samples_v1", "seed": args.seed,
                 "selection_rule": "nearest quantile over all true train rows x 17 prompts; first deterministic matching (row,prompt); mixed: NumPy RandomState(42) 128 unique rows then uniform prompt IDs",
                 "a_data_identity": a_identity, "source_identity": report["original_source"],
                 "tokenizer_path": str(output / "tokenizer_a"), "train_representatives": selections,
                 "train_mixed": mixed,
                 "test_prompt0_representatives": choose_samples(a_splits["test"],
                     {key: value[:, :1] for key, value in matrices["test"].items()}, np),
                 "test_all_prompt_representatives": choose_samples(a_splits["test"], matrices["test"], np),
                 "fixed_conditions": {"tasks": "seqrec", "train_prompt_sample_num": "1", "max_his_len": args.max_his_len,
                                      "his_sep": args.his_sep, "add_prefix": args.add_prefix,
                                      "only_train_response": True, "model_max_length": args.model_max_length}}
        write_json(output / "probe_samples.json", probe)
        arrays = {f"{mode}_{kind}_lengths": values for mode, matrix in matrices.items() for kind, values in matrix.items()}
        arrays.update({f"{mode}_history_lengths": np.asarray([r["history_len"] for r in a_splits[mode]], dtype=np.int16) for mode in a_splits})
        arrays.update({"valid_prompt_ids_seed42": valid_prompt_ids.astype(np.int16),
                       "train_user_ordinals": np.asarray([r["user_ordinal"] for r in a_splits["train"]], dtype=np.int32),
                       "train_target_positions": np.asarray([r["target_position"] for r in a_splits["train"]], dtype=np.int32)})
        np.savez_compressed(output / "length_rows.npz", **arrays)
        max_full = report["length_measurement"]["conditional_exhaustive_max_full_length"]
        report["truncation"] = {"model_max_length": args.model_max_length, "maximum_possible_full_length": max_full,
                                 "any_training_or_validation_truncation": max_full > args.model_max_length}
        if max_full > args.model_max_length:
            raise AssertionError("Observed length classes exceed the configured maximum; mask audit requires explicit truncation analysis")
        report["budget_counts"] = {"train_rows_per_epoch": len(a_splits["train"]),
                                    "valid_two_prompt_rows_per_epoch": len(valid_pairs),
                                    "test_prompt0_rows": len(a_splits["test"]),
                                    "test_all_17_prompt_rows": len(a_splits["test"]) * len(prompts),
                                    "epochs": 4, "global_batch": 64,
                                    "ceil_updates_per_epoch_at_batch64": (len(a_splits["train"]) + 63) // 64,
                                    "four_epoch_ceil_updates": 4 * ((len(a_splits["train"]) + 63) // 64),
                                    "update_rounding_note": "Exact Trainer/DeepSpeed terminal accumulation behavior must be bound to the runtime versions; ceil here is the conservative work estimate."}
        report["status"] = "PASS"
    except Exception as error:
        import traceback
        report["status"] = "PARTIAL" if isinstance(error, (TimeoutError, MemoryError)) else "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["cpu_process_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["output_files"] = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha_file(p)}
                                  for p in output.rglob("*") if p.is_file() and p.name != "audit.json"}
        write_json(output / "audit.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
