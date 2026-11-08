#!/usr/bin/env python3
"""Bounded CPU audit of real LC-Rec Instruments data and official supervision.

No model is constructed; no training, generation, evaluation metric or weight
reload is attempted. This script imports the supplied official source directly.
The full official six-task dataset is constructed before diagnostic selection.
"""
import argparse
import bisect
import collections
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time
import traceback


TASKS = ["seqrec", "item2index", "index2item", "fusionseqrec", "itemsearch", "preferenceobtain"]
SAMPLE_NUMS = [0, 0, 0, 20000, 0, 0]
DATA_FILES = ["Instruments.inter.json", "Instruments.item.json", "Instruments.index.json",
              "Instruments.user.json", "Instruments.item2id", "Instruments.user2id"]
EXPECTED_TOKENIZER_MODEL_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
EXPECTED_MODEL_REVISION = "4782ad278652c7c71b72204d462d6d01eaaf7549"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_record(value, limit=8000):
    value = str(value)
    return {"chars": len(value), "sha256_utf8": hashlib.sha256(value.encode()).hexdigest(),
            "text": value[:limit], "text_is_complete": len(value) <= limit}


def numeric_key(value):
    value = str(value)
    return (0, int(value), value) if value.isdecimal() else (1, 0, value)


def summarize_numbers(values):
    values = list(values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None}


def audit_assets(data_dir, report, check):
    loaded = {}
    files = []
    for name in DATA_FILES:
        path = data_dir / name
        entry = {"name": name, "path": str(path), "exists": path.is_file()}
        if not path.is_file():
            check("asset_exists:" + name, False)
            files.append(entry)
            continue
        entry.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        try:
            if name.endswith(".json"):
                value = json.loads(path.read_text())
                check("asset_mapping:" + name, isinstance(value, dict), type=type(value).__name__)
            else:
                rows = []
                for line in path.read_text().splitlines():
                    if line.strip():
                        original, remapped = line.rsplit("\t", 1)
                        rows.append((original, remapped))
                value = rows
            loaded[name] = value
            entry["records"] = len(value)
            check("asset_parse:" + name, True)
        except Exception as exc:
            entry["error"] = repr(exc)
            check("asset_parse:" + name, False, error=repr(exc))
        files.append(entry)
    report["assets"] = files
    inter = loaded.get("Instruments.inter.json", {})
    item = loaded.get("Instruments.item.json", {})
    index = loaded.get("Instruments.index.json", {})
    user = loaded.get("Instruments.user.json", {})
    if not all(isinstance(v, dict) for v in [inter, item, index, user]):
        return loaded
    item_axis, index_axis = set(item), set(index)
    interaction_axis = {str(i) for seq in inter.values() for i in seq}
    check("item_index_axis_equal", item_axis == index_axis,
          item_count=len(item_axis), index_count=len(index_axis),
          item_without_index=sorted(item_axis - index_axis, key=numeric_key)[:20],
          index_without_metadata=sorted(index_axis - item_axis, key=numeric_key)[:20])
    check("interaction_items_have_metadata_and_sid", interaction_axis <= (item_axis & index_axis),
          distinct_interaction_items=len(interaction_axis),
          absent=sorted(interaction_axis - (item_axis & index_axis), key=numeric_key)[:20])
    check("interactions_minimum_length", bool(inter) and all(len(seq) >= 4 for seq in inter.values()),
          lengths=summarize_numbers(map(len, inter.values())))
    for filename, target_axis in [("Instruments.item2id", item_axis), ("Instruments.user2id", set(inter))]:
        rows = loaded.get(filename, [])
        remapped = [row[1] for row in rows]
        original = [row[0] for row in rows]
        check("mapping_axis:" + filename, bool(rows) and set(remapped) == target_axis,
              entries=len(rows), distinct_remapped=len(set(remapped)), target_count=len(target_axis),
              absent_remapped=sorted(target_axis - set(remapped), key=numeric_key)[:20],
              extra_remapped=sorted(set(remapped) - target_axis, key=numeric_key)[:20])
        check("mapping_bijection:" + filename,
              len(original) == len(set(original)) and len(remapped) == len(set(remapped)))
    field_counts = collections.Counter()
    field_types = collections.defaultdict(collections.Counter)
    empty_fields = collections.Counter()
    malformed_items = []
    for iid, features in item.items():
        if not isinstance(features, dict):
            malformed_items.append(iid)
            continue
        for key, value in features.items():
            field_counts[key] += 1
            field_types[key][type(value).__name__] += 1
            if value is None or value == "" or value == []:
                empty_fields[key] += 1
        if not all(isinstance(features.get(key), str) for key in ["title", "description"]):
            malformed_items.append(iid)
    check("metadata_title_description_types", not malformed_items,
          malformed_count=len(malformed_items), examples=malformed_items[:20])
    report["metadata_fields"] = {"field_counts": dict(field_counts),
        "types": {k: dict(v) for k, v in field_types.items()}, "empty_counts": dict(empty_fields),
        "description_chars": summarize_numbers(len(v.get("description", "")) for v in item.values() if isinstance(v, dict)),
        "examples": [{"item_id": iid, "fields": {k: text_record(v, 1000) for k, v in item[iid].items()}}
                     for iid in sorted(item, key=numeric_key)[:2] if isinstance(item[iid], dict)],
        "meaning": "title and description are distinct official task fields; brand/categories retain their author-provided meanings and are not relabelled as description"}
    inverse = collections.defaultdict(list)
    lengths = collections.Counter()
    positions = collections.defaultdict(set)
    malformed_codes = []
    for iid, code in index.items():
        lengths[len(code)] += 1
        if isinstance(code, list) and all(isinstance(t, str) for t in code):
            inverse["".join(code)].append(iid)
            for pos, token in enumerate(code):
                positions[pos].add(token)
            good = len(code) == 4 and all(re.fullmatch(r"<" + "abcd"[j] + r"_(\d+)>", token)
                and 0 <= int(re.search(r"_(\d+)>", token).group(1)) < 256 for j, token in enumerate(code))
        else:
            good = False
        if not good:
            malformed_codes.append(iid)
    check("sid_four_official_groups", bool(index) and not malformed_codes,
          lengths=dict(lengths), malformed_count=len(malformed_codes), examples=malformed_codes[:20])
    duplicate_groups = {sid: ids for sid, ids in inverse.items() if len(ids) > 1}
    report["sid_catalog"] = {"items": len(index), "canonical_sid_count": len(inverse),
        "sid_length_counts": dict(lengths), "tokens_by_position": {str(k): sorted(v) for k, v in positions.items()},
        "distinct_tokens": len({t for values in positions.values() for t in values}),
        "duplicate_groups": len(duplicate_groups), "extra_items_due_to_duplicates": len(index) - len(inverse),
        "duplicate_examples": [{"canonical_sid": sid, "item_ids": ids} for sid, ids in list(duplicate_groups.items())[:20]],
        "collision_policy": "Preserve author SID mapping; report residual duplicates; never append another SID position or choose an arbitrary unique item."}
    explicit = user.get("user_explicit_preference", {})
    intentions = user.get("user_vague_intention", {})
    train_intentions = intentions.get("train", {}) if isinstance(intentions, dict) else {}
    test_intentions = intentions.get("test", {}) if isinstance(intentions, dict) else {}
    check("user_auxiliary_sections", isinstance(explicit, dict) and bool(explicit)
          and isinstance(train_intentions, dict) and bool(train_intentions),
          explicit_count=len(explicit), train_intent_count=len(train_intentions), test_intent_count=len(test_intentions))
    aux_errors = []
    test_errors = []
    for uid, preferences in explicit.items():
        reasons = []
        if not isinstance(preferences, list) or not preferences or not all(isinstance(x, str) and x.strip() for x in preferences):
            reasons.append("explicit_preferences_not_nonempty_string_list")
        if uid not in inter or uid not in train_intentions:
            reasons.append("missing_interaction_or_training_intention")
        else:
            seq = list(map(str, inter[uid]))
            intent = train_intentions[uid]
            if str(intent.get("item")) != seq[-3]:
                reasons.append("train_intent_target_not_inter_minus3")
            if list(map(str, intent.get("inters", []))) != seq[:-3]:
                reasons.append("train_intent_history_not_inter_before_minus3")
            queries = intent.get("querys")
            if not isinstance(queries, list) or len(queries) != 2 or not all(isinstance(x, str) and x.strip() for x in queries):
                reasons.append("querys_not_two_nonempty_strings")
        if reasons:
            aux_errors.append({"uid": uid, "reasons": reasons})
    for uid, intent in test_intentions.items():
        if uid not in inter:
            test_errors.append({"uid": uid, "reason": "no_interaction"})
        elif str(intent.get("item")) != str(inter[uid][-1]) or list(map(str, intent.get("inters", []))) != list(map(str, inter[uid][:-1])):
            test_errors.append({"uid": uid, "reason": "test_target_or_history_mismatch"})
    check("training_auxiliary_temporal_pairing", bool(explicit) and not aux_errors,
          checked_users=len(explicit), mismatch_count=len(aux_errors), examples=aux_errors[:20])
    check("available_test_intention_pairing", not test_errors,
          checked_users=len(test_intentions), mismatch_count=len(test_errors), examples=test_errors[:20],
          note="An empty auxiliary test section is reported as empty; it is not consumed by this six-task training path or SeqRec test.")
    report["temporal_protocol"] = {"seqrec_train_prefix_rows": sum(max(0, len(seq) - 3) for seq in inter.values()),
        "train_visible_items": "full_interactions[:-2]; all prefix-next pairs starting at prefix length 1",
        "valid_target": "full_interactions[-2]", "valid_history": "full_interactions[:-2]",
        "test_target": "full_interactions[-1]", "test_history": "full_interactions[:-1]",
        "intention_train_target": "full_interactions[-3]", "intention_train_history": "full_interactions[:-3]",
        "preference_history": "full_interactions[:-3]", "max_history_unit": "20 items, then independent 2048-token truncation",
        "auxiliary_train_user_coverage": len(explicit) / len(inter) if inter else None}
    return loaded


def source_identity(source):
    result = []
    for name in ["utils.py", "data.py", "prompt.py", "collator.py", "finetune.py", "test.py", "evaluate.py"]:
        path = source / name
        result.append({"path": name, "exists": path.is_file(),
                       "sha256": sha256_file(path) if path.is_file() else None})
    return result


def prepared_record(task, record):
    record = dict(record)
    if task in ["itemsearch", "preferenceobtain"]:
        preferences = record["explicit_preferences"]
        record["explicit_preference"] = max(preferences, key=len)
    if task == "itemsearch":
        record["query"] = max([record["user_related_intention"], record["item_related_intention"]], key=len)
    return record


def processed_records(task, dataset):
    return getattr(dataset, {"seqrec": "inter_data", "fusionseqrec": "inter_data", "item2index": "feat_data",
                             "index2item": "feat_data", "itemsearch": "search_data", "preferenceobtain": "preference_data"}[task])


def select_diagnostic_rows(task, dataset, seed):
    records = processed_records(task, dataset)
    longest = None
    shortest = None
    scan_errors = []
    rendered_count = 0
    for idx, original in enumerate(records):
        try:
            record = prepared_record(task, original)
            for prompt_id, prompt in enumerate(dataset.prompts):
                input_text, full_text = dataset._get_text_data(record, prompt)
                entry = {"dataset_row": idx, "prompt_id": prompt_id,
                         "input_ids": input_text, "labels": full_text,
                         "rendered_chars": len(full_text), "record": record,
                         "selection_kind": "directed_official_render"}
                rendered_count += 1
                if longest is None or len(full_text) > longest["rendered_chars"]:
                    longest = entry
                if shortest is None or len(full_text) < shortest["rendered_chars"]:
                    shortest = entry
        except Exception as exc:
            if len(scan_errors) < 20:
                scan_errors.append({"dataset_row": idx, "error": repr(exc)})
    chosen = []
    if longest:
        longest["selection_reason"] = "maximum rendered character count across all official processed records and all task prompts; longest real preference/query components where applicable"
        chosen.append(longest)
    if records:
        state = random.getstate()
        random.seed(seed)
        try:
            actual = dataset[0]
            normal = {"dataset_row": 0, "prompt_id": None, **actual,
                      "record": dict(records[0]), "selection_kind": "official___getitem__",
                      "selection_reason": "first official processed row with task RNG temporarily set to the declared seed"}
            for prompt_id, prompt in enumerate(dataset.prompts):
                input_text, full_text = dataset._get_text_data(normal["record"], prompt)
                if input_text == actual["input_ids"] and full_text == actual["labels"]:
                    normal["prompt_id"] = prompt_id
                    break
            chosen.append(normal)
        finally:
            random.setstate(state)
    if shortest:
        shortest["selection_reason"] = "minimum rendered character count in the same processed-record/template scan, to exercise real padding alongside the long path"
        chosen.append(shortest)
    deduplicated = []
    seen = set()
    for row in chosen:
        identity = (row["dataset_row"], row["prompt_id"], row["input_ids"], row["labels"])
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(row)
    return deduplicated[:3], {"processed_records_scanned": len(records), "rendered_variants_scanned": rendered_count,
        "scan_errors": scan_errors, "length_selection_unit": "Unicode characters, not tokenizer tokens",
        "resource_claim": "The character-longest candidate is tokenized and reported; this does not prove it is the global tokenizer-longest input or establish GPU fit."}


def origin_for(task, row, loaded, args):
    inter = loaded["Instruments.inter.json"]
    index = loaded["Instruments.index.json"]
    metadata = loaded["Instruments.item.json"]
    dataset_row = row["dataset_row"]
    record = row["record"]
    if task in ["item2index", "index2item"]:
        iid = list(metadata)[dataset_row]
        return {"item_id": iid, "canonical_sid": "".join(index[iid])}
    if task in ["itemsearch", "preferenceobtain"]:
        uid = list(loaded["Instruments.user.json"]["user_explicit_preference"])[dataset_row]
        intent = loaded["Instruments.user.json"]["user_vague_intention"]["train"][uid]
        return {"user_id": uid, "history_item_ids": list(map(str, inter[uid][:-3]))[-args.max_his_len:],
                "intention_target_item_id": str(intent["item"]),
                "intention_target_canonical_sid": "".join(index[str(intent["item"])])}
    if task == "seqrec":
        cumulative, users = [], []
        total = 0
        for uid, seq in inter.items():
            total += max(0, len(seq) - 3)
            cumulative.append(total)
            users.append(uid)
        user_pos = bisect.bisect_right(cumulative, dataset_row)
        previous = cumulative[user_pos - 1] if user_pos else 0
        target_pos = dataset_row - previous + 1
        uid = users[user_pos]
        seq = list(map(str, inter[uid]))
        return {"user_id": uid, "target_position_in_full_interaction": target_pos,
                "target_item_id": seq[target_pos], "target_canonical_sid": "".join(index[seq[target_pos]]),
                "history_item_ids": seq[max(0, target_pos - args.max_his_len):target_pos]}
    matches = []
    for uid, raw_seq in inter.items():
        seq = list(map(str, raw_seq))
        for target_pos in range(1, len(seq) - 2):
            iid = seq[target_pos]
            if "".join(index[iid]) != record["item"]:
                continue
            history = seq[max(0, target_pos - args.max_his_len):target_pos]
            if args.his_sep.join("".join(index[i]) for i in history) != record["inters"]:
                continue
            if metadata[iid]["title"].strip().strip(".!?,;:`") != record["title"]:
                continue
            matches.append({"user_id": uid, "target_position_in_full_interaction": target_pos,
                            "target_item_id": iid, "target_canonical_sid": record["item"], "history_item_ids": history})
            if len(matches) >= 5:
                return {"matching_origins": matches, "matching_origins_capped_at": 5}
    return {"matching_origins": matches, "note": "Fusion sampling removes source row IDs; match against real training prefixes, retaining ambiguity."}


def inspect_collated(rows, collator, tokenizer, torch, check, label):
    if not rows:
        check(label + ":has_real_rows", False)
        return []
    batch = collator([{k: row[k] for k in ["input_ids", "labels"]} for row in rows])
    prompts = tokenizer([row["input_ids"] for row in rows], padding="longest", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt")
    width = batch["input_ids"].shape[1]
    result = []
    for pos, row in enumerate(rows):
        ids = batch["input_ids"][pos]
        labels = batch["labels"][pos]
        attention = batch["attention_mask"][pos]
        prompt_ids = prompts["input_ids"][pos][prompts["attention_mask"][pos].bool()]
        prompt_n = len(prompt_ids)
        full_raw = tokenizer.encode(row["labels"] + tokenizer.eos_token, truncation=False)
        prompt_raw = tokenizer.encode(row["input_ids"], truncation=False)
        expected = ids.clone()
        expected[ids == tokenizer.pad_token_id] = -100
        expected[:min(prompt_n, width)] = -100
        mask_ok = bool(torch.equal(labels, expected))
        check(label + ":official_response_mask:" + str(pos), mask_ok)
        supervised = labels != -100
        check(label + ":nonempty_real_supervision:" + str(pos), bool(supervised.any()),
              untruncated_prompt_tokens=len(prompt_raw), untruncated_full_tokens=len(full_raw),
              note="Failure is retained; the script does not shorten or replace a long real sample.")
        prefix_n = min(prompt_n, width)
        prefix_match = bool(torch.equal(ids[:prefix_n], prompt_ids[:prefix_n]))
        check(label + ":prompt_full_token_prefix:" + str(pos), prefix_match)
        nonpad = ids[attention.bool()]
        if len(nonpad):
            check(label + ":bos:" + str(pos), int(nonpad[0]) == tokenizer.bos_token_id)
        truncated = len(full_raw) > tokenizer.model_max_length
        if not truncated and len(nonpad):
            check(label + ":appended_eos:" + str(pos), int(nonpad[-1]) == tokenizer.eos_token_id)
        check(label + ":padding_ignored:" + str(pos), bool(torch.all(labels[attention == 0] == -100)))
        result.append({"dataset_row": row.get("dataset_row"), "prompt_id": row.get("prompt_id"),
            "selection_kind": row.get("selection_kind"), "selection_reason": row.get("selection_reason"),
            "origin": row.get("origin"), "input_text": text_record(row["input_ids"]),
            "full_supervision_text_before_appended_eos": text_record(row["labels"]),
            "untruncated_prompt_tokens": len(prompt_raw), "untruncated_full_tokens_with_eos": len(full_raw),
            "max_total_tokens": tokenizer.model_max_length, "full_text_truncated": truncated,
            "history_item_limit": 20, "collated_total_width": width,
            "nonpadding_tokens": int(attention.sum()), "supervised_tokens": int(supervised.sum()),
            "padding_tokens": int((attention == 0).sum()),
            "unk_or_pad_id_in_nonpadding_positions": int(((ids == tokenizer.pad_token_id) & attention.bool()).sum()),
            "input_token_ids": ids.tolist(), "label_token_ids": labels.tolist(),
            "target_mask": supervised.to(dtype=torch.int8).tolist(), "attention_mask": attention.tolist(),
            "supervised_decoded_text": text_record(tokenizer.decode(ids[supervised].tolist(), skip_special_tokens=False)),
            "mask_matches_official_response_only_rule": mask_ok, "prompt_full_prefix_match": prefix_match})
    return result


def audit_runtime(cli, loaded, report, check):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(cli.source))
    import torch
    import numpy as np
    import transformers
    import sentencepiece
    official_utils = importlib.import_module("utils")
    official_data = importlib.import_module("data")
    official_collator = importlib.import_module("collator")
    official_prompt = importlib.import_module("prompt")
    modules = {"torch": torch, "numpy": np, "transformers": transformers, "sentencepiece": sentencepiece,
               "utils": official_utils, "data": official_data, "collator": official_collator, "prompt": official_prompt}
    report["environment"] = {"python_executable": sys.executable, "python_version": sys.version,
        "imports": {name: {"file": module.__file__, "version": getattr(module, "__version__", None)} for name, module in modules.items()},
        "cuda_visible_devices_for_this_process": "", "model_instantiated": False}
    for name in ["utils", "data", "collator", "prompt"]:
        check("official_import:" + name, Path(modules[name].__file__).resolve().parent == cli.source)
    parser = argparse.ArgumentParser(add_help=False)
    parser = official_utils.parse_global_args(parser)
    parser = official_utils.parse_train_args(parser)
    parser = official_utils.parse_dataset_args(parser)
    args = parser.parse_args([])
    for key, value in {"seed": cli.seed, "dataset": "Instruments", "data_path": str(cli.data_root),
            "base_model": str(cli.tokenizer), "max_his_len": 20, "model_max_length": 2048,
            "index_file": ".index.json", "only_train_response": True,
            "tasks": ",".join(TASKS), "train_prompt_sample_num": "1,1,1,1,1,1",
            "train_data_sample_num": "0,0,0,20000,0,0"}.items():
        setattr(args, key, value)
    report["official_dataset_effective_arguments"] = vars(args)
    report["scope"] = {"full_official_dataset_construction": True, "diagnostic_real_rows_per_task": "at most 3",
        "validation_real_rows_collated": "at most 32, zero loss computations", "generation_users_planned": 16,
        "selection_depends_on_metrics": False, "sampling_zero_semantics": "official data.py only samples when sample_num > 0; zero does not empty a task"}
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    datasets, valid_data = {}, None
    try:
        train_data, valid_data = official_utils.load_datasets(args)
        datasets = dict(zip(TASKS, train_data.datasets))
        report["official_full_loader"] = {"status": "PASS", "concat_rows": len(train_data),
            "task_rows": {task: len(ds) for task, ds in datasets.items()}, "full_validation_text_rows_constructed": len(valid_data),
            "full_validation_executed": False, "note": "Constructing official validation strings does not run validation loss or model inference."}
    except Exception as exc:
        check("official_full_loader", False, error=repr(exc))
        report["official_full_loader"] = {"status": "PARTIAL", "error": repr(exc),
            "fallback": "Each independently available task is diagnosed using the same official load_datasets API; fallback RNG sequence is not claimed identical to a successful full six-task construction."}
        for task, sample_num in zip(TASKS, SAMPLE_NUMS):
            diagnostic = argparse.Namespace(**vars(args))
            diagnostic.tasks = task
            diagnostic.train_prompt_sample_num = "1"
            diagnostic.train_data_sample_num = str(sample_num)
            random.seed(cli.seed)
            np.random.seed(cli.seed)
            try:
                part, validation = official_utils.load_datasets(diagnostic)
                datasets[task] = part.datasets[0]
                if valid_data is None:
                    valid_data = validation
            except Exception as task_exc:
                report.setdefault("task_loader_errors", {})[task] = repr(task_exc)
    check("all_six_real_task_datasets_available", set(datasets) == set(TASKS), available=list(datasets))
    # Compare the constructed official rows to real raw interaction slices, not
    # merely to names/comments in source or to their total lengths.
    inter = loaded.get("Instruments.inter.json", {})
    catalog = loaded.get("Instruments.index.json", {})
    if "seqrec" in datasets and inter and catalog:
        actual_rows = datasets["seqrec"].inter_data
        row_pos, mismatch_count, examples = 0, 0, []
        for uid, sequence in inter.items():
            sequence = list(map(str, sequence))
            for target_pos in range(1, len(sequence) - 2):
                history = sequence[max(0, target_pos - 20):target_pos]
                expected = {"item": "".join(catalog[sequence[target_pos]]),
                            "inters": args.his_sep.join("".join(catalog[i]) for i in history)}
                actual = actual_rows[row_pos] if row_pos < len(actual_rows) else None
                if actual is None or any(actual.get(k) != v for k, v in expected.items()):
                    mismatch_count += 1
                    if len(examples) < 10:
                        examples.append({"user_id": uid, "target_position": target_pos, "official_row": row_pos})
                row_pos += 1
        check("official_seqrec_every_training_prefix_matches_leave_last2", mismatch_count == 0 and row_pos == len(actual_rows),
              checked_rows=row_pos, official_rows=len(actual_rows), mismatches=mismatch_count, examples=examples)
    if valid_data is not None and inter and catalog:
        mismatches = []
        for pos, (uid, sequence) in enumerate(inter.items()):
            sequence = list(map(str, sequence))
            expected_item = "".join(catalog[sequence[-2]])
            expected_history = args.his_sep.join("".join(catalog[i]) for i in sequence[:-2][-20:])
            row = valid_data.inter_data[pos] if pos < len(valid_data.inter_data) else None
            if row is None or row.get("item") != expected_item or row.get("inters") != expected_history:
                mismatches.append(uid)
        check("official_validation_every_user_matches_penultimate_target", not mismatches and len(valid_data.inter_data) == len(inter),
              users_checked=len(inter), mismatches=len(mismatches), examples=mismatches[:20])
    if not loaded.get("Instruments.index.json"):
        report["tokenizer_audit"] = {"status": "BLOCKED", "reason": "real SID catalog unavailable"}
        return
    tokenizer_files = []
    for path in sorted(cli.tokenizer.iterdir()):
        if path.is_file() and path.name in ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "generation_config.json", "LICENSE", "README.md"]:
            tokenizer_files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    tokenizer_model = next((row for row in tokenizer_files if row["name"] == "tokenizer.model"), None)
    check("tokenizer_model_matches_fixed_hf_lfs_sha256",
          tokenizer_model is not None and tokenizer_model["sha256"] == EXPECTED_TOKENIZER_MODEL_SHA256,
          expected_revision=EXPECTED_MODEL_REVISION, expected_sha256=EXPECTED_TOKENIZER_MODEL_SHA256,
          actual_sha256=tokenizer_model["sha256"] if tokenizer_model else None)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(str(cli.tokenizer), local_files_only=True,
        model_max_length=args.model_max_length, padding_side="right")
    tokenizer.pad_token_id = 0
    base_len = len(tokenizer)
    check("original_llama_tokenizer_base_ids", base_len == 32000 and tokenizer.bos_token_id == 1
          and tokenizer.eos_token_id == 2 and tokenizer.unk_token_id == 0 and tokenizer.pad_token_id == 0)
    real_index = loaded["Instruments.index.json"]
    if "seqrec" in datasets:
        new_tokens = datasets["seqrec"].get_new_tokens()
        token_source = "official first SeqRecDataset.get_new_tokens()"
    else:
        new_tokens = sorted({t for code in real_index.values() for t in code})
        token_source = "same sorted real SID-token union; official first dataset unavailable, partial diagnostic only"
    added = tokenizer.add_tokens(new_tokens)
    mappings = [{"token": token, "id": tokenizer.convert_tokens_to_ids(token),
                 "without_special_ids": tokenizer.encode(token, add_special_tokens=False),
                 "default_ids": tokenizer.encode(token)} for token in new_tokens]
    check("all_sid_tokens_single_ids", bool(mappings) and all(row["without_special_ids"] == [row["id"]] for row in mappings))
    check("sid_default_tokenization_is_bos_then_single_sid", bool(mappings)
          and all(row["default_ids"] == [tokenizer.bos_token_id, row["id"]] for row in mappings))
    check("sid_ids_are_new_contiguous_rows", added == len(new_tokens)
          and sorted(row["id"] for row in mappings) == list(range(base_len, base_len + len(new_tokens))))
    report["tokenizer_audit"] = {"class": type(tokenizer).__name__, "path": str(cli.tokenizer),
        "base_vocabulary_size": base_len, "new_tokens_added": added, "expanded_vocabulary_size": len(tokenizer),
        "token_source": token_source, "sid_token_mapping": mappings, "files": tokenizer_files,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id, "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side, "model_max_length": tokenizer.model_max_length,
        "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token,
        "chat_template_used": False, "embedding_or_head_loaded": False,
        "initialization_and_gradient_status": "PENDING: token IDs only; no embedding/head weights or updates were inspected"}
    collator = official_collator.Collator(args, tokenizer)
    report["task_supervision"] = {}
    for task in TASKS:
        if task not in datasets:
            report["task_supervision"][task] = {"status": "BLOCKED", "reason": "official real task dataset construction failed"}
            continue
        try:
            print(json.dumps({"event": "task_cpu_audit_start", "task": task}), flush=True)
            dataset = datasets[task]
            check("nonempty_task:" + task, len(dataset) > 0, rows=len(dataset))
            selected, scan = select_diagnostic_rows(task, dataset, cli.seed)
            check("all_official_templates_render:" + task, not scan["scan_errors"], examples=scan["scan_errors"])
            for row in selected:
                row["origin"] = origin_for(task, row, loaded, args)
                history = row["origin"].get("history_item_ids")
                if history is not None:
                    check("history_item_limit:" + task + ":" + str(row["dataset_row"]), len(history) <= 20)
            before = len(report["checks"])
            samples = inspect_collated(selected, collator, tokenizer, torch, check, task)
            checks = report["checks"][before:]
            report["task_supervision"][task] = {"status": "PASS" if samples and all(x["pass"] for x in checks) and not scan["scan_errors"] else "PARTIAL",
                "official_dataset_rows": len(dataset), "scan": scan, "samples": samples,
                "functional_coverage_only": True, "not_a_gpu_resource_preflight": True}
            print(json.dumps({"event": "task_cpu_audit_done", "task": task,
                              "status": report["task_supervision"][task]["status"], "samples": len(samples)}), flush=True)
        except Exception as exc:
            check("task_supervision:" + task, False, error=repr(exc))
            report["task_supervision"][task] = {"status": "PARTIAL", "error": repr(exc), "traceback": traceback.format_exc()}
    inter = loaded.get("Instruments.inter.json", {})
    selected_uids = sorted(inter, key=numeric_key)[:16]
    report["fixed_user_selection"] = {"rule": "first 16 real user IDs in numeric order, before any model result; shared by the future bounded generation check",
        "user_ids": selected_uids, "contains_non_numeric_ids": any(not uid.isdecimal() for uid in selected_uids),
        "seed": cli.seed, "metric_based_selection": False}
    if valid_data is not None:
        try:
            all_uid_order = list(inter)
            per_uid = valid_data.prompt_sample_num
            fixed_validation = []
            for uid in selected_uids:
                user_pos = all_uid_order.index(uid)
                for offset in range(per_uid):
                    if len(fixed_validation) >= 32:
                        break
                    row_idx = user_pos * per_uid + offset
                    row = dict(valid_data[row_idx])
                    row.update(dataset_row=row_idx, user_id=uid, prompt_id=None,
                               selection_kind="fixed_validation_real_row", selection_reason="numeric-first fixed user, official sampled validation templates")
                    for prompt_id, prompt in enumerate(valid_data.prompts):
                        source_record = valid_data.inter_data[user_pos]
                        expected_input, expected_full = valid_data._get_text_data(source_record, prompt)
                        if row["input_ids"] == expected_input and row["labels"] == expected_full:
                            row["prompt_id"] = prompt_id
                            break
                    row["origin"] = {"user_id": uid, "target_item_id": str(inter[uid][-2]),
                        "target_canonical_sid": "".join(real_index[str(inter[uid][-2])]),
                        "history_item_ids": list(map(str, inter[uid][:-2]))[-20:]}
                    fixed_validation.append(row)
            validation_samples = inspect_collated(fixed_validation, collator, tokenizer, torch, check, "fixed_validation")
            report["fixed_validation"] = {"rows_collated": len(validation_samples), "loss_computations": 0,
                "selection": [{"user_id": row["origin"]["user_id"], "official_row": row["dataset_row"],
                    "prompt_id": row["prompt_id"], "target_item_id": row["origin"]["target_item_id"],
                    "untruncated_full_tokens": row["untruncated_full_tokens_with_eos"], "supervised_tokens": row["supervised_tokens"]}
                    for row in validation_samples],
                "token_evidence_first_two": validation_samples[:2], "full_validation_executed": False}
            check("validation_collation_budget", 0 < len(validation_samples) <= 32)
        except Exception as exc:
            check("fixed_validation", False, error=repr(exc))
            report["fixed_validation"] = {"status": "PARTIAL", "error": repr(exc), "loss_computations": 0}
    if "seqrec" in datasets and inter:
        try:
            test_data = official_data.SeqRecDataset(args, mode="test", prompt_sample_num=1, sample_num=0)
            prefix_fn = test_data.get_prefix_allowed_tokens_fn(tokenizer)
            inverse = collections.defaultdict(list)
            canonical_codes = set()
            for iid, code in real_index.items():
                inverse["".join(code)].append(iid)
                canonical_codes.add(tuple(code))
            allowed_expected = {pos: {tokenizer.convert_tokens_to_ids(code[pos]) for code in real_index.values()} for pos in range(4)}
            allowed_expected[4] = {tokenizer.eos_token_id}
            check("official_per_position_catalog_token_sets", all(set(test_data.allowed_tokens.get(pos, [])) == tokens for pos, tokens in allowed_expected.items()))
            fixed_test = []
            constraint_calls = 0
            constraint_mismatches = []
            for uid in selected_uids:
                record = {"item": "".join(real_index[str(inter[uid][-1])]),
                    "inters": args.his_sep.join("".join(real_index[str(i)]) for i in inter[uid][:-1][-20:])}
                template_rows = []
                code_ids = [tokenizer.convert_tokens_to_ids(token) for token in real_index[str(inter[uid][-1])]]
                for prompt_id, prompt in enumerate(official_prompt.all_prompt["seqrec"]):
                    input_text, target = test_data._get_text_data(record, prompt)
                    prefix_ids = tokenizer.encode(input_text)
                    for position in range(5):
                        suffix = code_ids[:position]
                        actual_allowed = set(prefix_fn(0, torch.tensor(prefix_ids + suffix, dtype=torch.long)))
                        constraint_calls += 1
                        if actual_allowed != allowed_expected[position]:
                            constraint_mismatches.append({"user_id": uid, "prompt_id": prompt_id, "position": position})
                    template_rows.append({"prompt_id": prompt_id, "input": text_record(input_text, 2000),
                                          "input_token_count": len(prefix_ids), "target_canonical_sid": target})
                fixed_test.append({"user_id": uid, "target_item_id": str(inter[uid][-1]),
                    "target_canonical_sid": record["item"], "canonical_sid_item_ids": inverse[record["item"]],
                    "history_item_ids": list(map(str, inter[uid][:-1]))[-20:], "templates": template_rows})
            check("official_constraint_callback_real_prompt_positions", not constraint_mismatches,
                  calls=constraint_calls, mismatch_examples=constraint_mismatches[:20])
            hybrid = None
            base_code = tuple(next(iter(real_index.values())))
            for position in range(4):
                for token_id in sorted(allowed_expected[position]):
                    candidate = list(base_code)
                    candidate[position] = tokenizer.convert_ids_to_tokens(token_id)
                    if tuple(candidate) not in canonical_codes:
                        hybrid = {"sid": "".join(candidate), "position_token_ids": [tokenizer.convert_tokens_to_ids(t) for t in candidate],
                                  "every_position_allowed": True, "canonical_catalog_member": False}
                        break
                if hybrid:
                    break
            report["canonical_and_constraints"] = {"status": "STATIC_AND_CONSTRUCTED_CHECK_ONLY", "constraint_callback_calls": constraint_calls,
                "method": "Call the unmodified official callback on real prompt token IDs plus real canonical target-token prefixes; never call model.generate",
                "allowed_counts_by_position": {str(k): len(v) for k, v in allowed_expected.items()},
                "canonical_inverse_policy": "SID maps to all matching real item IDs; residual collision ambiguity is retained",
                "official_constraint_is_full_sid_trie": False,
                "explanation": "Official callback constrains each SID position to that position's catalog vocabulary; it does not enforce that the completed four-token combination is an observed canonical item SID.",
                "constructed_noncanonical_combination_probe": hybrid,
                "probe_is_a_model_output": False, "beam_search_executed": False, "filtering_or_dedup_changed": False,
                "fixed_test_users": fixed_test, "seqrec_template_count": len(official_prompt.all_prompt["seqrec"])}
        except Exception as exc:
            check("canonical_constraint_check", False, error=repr(exc))
            report["canonical_and_constraints"] = {"status": "PARTIAL", "error": repr(exc), "generation_executed": False}
    report["environment"]["cuda_initialized_at_end"] = torch.cuda.is_initialized()
    check("cpu_only_no_cuda_context", not torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official vendor source root containing utils.py/data.py/collator.py/prompt.py")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing Instruments/")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local fixed-revision tokenizer directory; no network fallback")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()
    if cli.seed != 42:
        parser.error("This preparation audit is authorized only for seed=42")
    cli.source, cli.data_root, cli.tokenizer = (p.resolve() for p in [cli.source, cli.data_root, cli.tokenizer])
    started = time.monotonic()
    report = {"schema": "lc-rec-real-data-cpu-audit-v1", "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_sha256": sha256_file(__file__), "actual_argv": sys.argv,
        "cwd": str(Path.cwd()), "seed": cli.seed, "checks": [], "runtime_errors": [],
        "not_performed": {stage: {"status": "PENDING_NOT_EXECUTED", "reason": "Outside CPU data/tokenizer audit; requires separately authorized and resource-feasible execution"}
            for stage in ["model_weight_loading", "embedding_head_initialization", "gradients", "optimizer_updates", "validation_loss",
                          "checkpoint_save", "checkpoint_reload", "model_generation", "recommendation_metrics", "optimizer_resume"]},
        "formal_training_runs": 0, "optimizer_updates": 0, "generated_users": 0}
    def check(name, passed, **details):
        report["checks"].append({"name": name, "pass": bool(passed), **details})
    try:
        report["source_identity"] = source_identity(cli.source)
        check("official_source_required_files", all(row["exists"] for row in report["source_identity"]))
        loaded = audit_assets(cli.data_root / "Instruments", report, check)
        audit_runtime(cli, loaded, report, check)
    except Exception as exc:
        report["runtime_errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        check("runtime_completed", False, error=repr(exc))
    failures = [row for row in report["checks"] if not row["pass"]]
    report["cpu_data_tokenizer_supervision_status"] = "PASS" if report["checks"] and not failures else "PARTIAL"
    report["overall_reproduction_status"] = "PARTIAL"
    report["status_boundary"] = "CPU checks can pass only within their recorded real-data coverage. Model correctness, GPU fit, saving/reload, generation and paper reproduction remain unverified."
    report["failed_check_names"] = [row["name"] for row in failures]
    report["wall_seconds"] = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["cpu_process_resource_usage"] = {"elapsed_wall_seconds": report["wall_seconds"],
        "ru_maxrss": usage.ru_maxrss, "ru_maxrss_unit": "KiB on Linux; bytes on macOS",
        "peak_cpu_rss_bytes": int(usage.ru_maxrss * 1024) if sys.platform.startswith("linux") else int(usage.ru_maxrss),
        "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
        "measurement_scope": "This CPU audit process only, including imported libraries and official dataset construction; GPU memory is not measured.",
        "gpu_resource_preflight_executed": False}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cli.output.with_name(cli.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, cli.output)
    print(json.dumps({"output": str(cli.output), "cpu_data_tokenizer_supervision_status": report["cpu_data_tokenizer_supervision_status"],
        "failed_checks": len(failures), "wall_seconds": report["wall_seconds"], "formal_training_runs": 0,
        "optimizer_updates": 0, "overall_reproduction_status": "PARTIAL"}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
