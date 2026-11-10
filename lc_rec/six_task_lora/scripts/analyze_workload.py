#!/usr/bin/env python3
"""Measure one seed42 realization of official LC-Rec six-task CPU supervision.

No model weights are read, no CUDA context is initialized, and no formal
training corpus is filtered. Every training record is obtained by calling the
unmodified official Dataset.__getitem__ once in task/local-index order; the
official Collator is then actually executed. Auxiliary mismatches remain in
the distribution and are explicitly limited to diagnostic performance use.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import importlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback

from data_support import (TASKS, EXPECTED_ROWS, EXPECTED_TOKENIZER_SHA256, Origins,
                          ObservedRandom, file_sha256, text_sha256, json_dump,
                          largest_remainder, number_summary, padding_summary)


def rng_snapshot(np, torch):
    n = np.random.get_state()
    return {"python_random": random.getstate(),
            "numpy_random": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch_cpu_random": torch.get_rng_state().tolist()}


def row_metrics(row, batch, pos, tokenizer, raw_full, raw_input, answer_ids, torch):
    ids, labels, mask = (batch[key][pos] for key in ["input_ids", "labels", "attention_mask"])
    input_n = min(len(raw_input), tokenizer.model_max_length)
    active = labels != -100
    expected = ids.clone()
    expected[expected == tokenizer.pad_token_id] = -100
    expected[:input_n] = -100
    assert torch.equal(labels, expected), "Official collator response-only mask disagrees with recorded raw lengths"
    assert bool(torch.all(labels[mask == 0] == -100)), "Padding received supervision"
    useful = int(mask.sum())
    supervised = labels[active].tolist()
    return {"input_raw_tokens": len(raw_input), "answer_standalone_tokens": len(answer_ids),
            "full_raw_tokens_with_eos": len(raw_full), "input_tokens": input_n,
            "full_tokens": useful, "effective_label_tokens": int(active.sum()),
            "shift_effective_label_tokens": int((labels[1:] != -100).sum()),
            "raw_response_suffix_tokens_with_eos": max(0, len(raw_full) - len(raw_input)),
            "input_truncated": len(raw_input) > tokenizer.model_max_length,
            "full_truncated": len(raw_full) > tokenizer.model_max_length,
            "answer_text_empty": row["labels"] == row["input_ids"],
            "eos_only_supervision": bool(supervised) and all(token == tokenizer.eos_token_id for token in supervised),
            "zero_supervision": not supervised,
            "prompt_full_token_prefix_match": raw_full[:input_n] == raw_input[:input_n],
            "starts_with_bos": bool(useful) and int(ids[0]) == tokenizer.bos_token_id,
            "ends_with_eos_after_truncation": bool(useful) and int(ids[useful - 1]) == tokenizer.eos_token_id,
            "unk_or_pad_ids_inside_nonpadding": int(((ids == tokenizer.pad_token_id) & mask.bool()).sum()),
            "analysis_chunk_width": int(ids.numel()), "analysis_chunk_padding_tokens": int(ids.numel()) - useful,
            "analysis_chunk_padding_ratio": 1 - useful / ids.numel(),
            "input_sha256_utf8": text_sha256(row["input_ids"]),
            "full_sha256_utf8": text_sha256(row["labels"])}


def describe_subset(metadata, indices, np):
    fields = ["input_raw_tokens", "answer_standalone_tokens", "full_raw_tokens_with_eos", "input_tokens",
              "full_tokens", "effective_label_tokens", "shift_effective_label_tokens",
              "raw_response_suffix_tokens_with_eos", "analysis_chunk_padding_ratio"]
    result = {key: number_summary([metadata[i][key] for i in indices], np) for key in fields}
    flags = ["input_truncated", "full_truncated", "answer_text_empty", "eos_only_supervision", "zero_supervision",
             "prompt_full_token_prefix_match", "starts_with_bos", "ends_with_eos_after_truncation",
             "technical_update_eligible", "diagnostic_only"]
    result["counts"] = {key: sum(bool(metadata[i][key]) for i in indices) for key in flags}
    result["prompt_id_counts"] = dict(collections.Counter(metadata[i]["prompt_id"] for i in indices))
    lengths = [metadata[i]["full_tokens"] for i in indices]
    result["padding_diagnostic"] = {str(micro): {
        "task_index_order": padding_summary(lengths, micro, np),
        "length_bucketed": padding_summary(lengths, micro, np, sort=True)} for micro in [1, 2, 4, 8]}
    return result


def select_plans(metadata, ranges, np):
    rng = np.random.RandomState(42)
    plans = {"task_representatives": {}, "task_technical_representatives": {}}
    reasons = collections.defaultdict(list)
    for task in TASKS:
        start, end = ranges[task]
        candidates = list(range(start, end))
        for key, only_safe in [("task_representatives", False), ("task_technical_representatives", True)]:
            legal = [i for i in candidates if not metadata[i]["full_truncated"]
                     and metadata[i]["shift_effective_label_tokens"] > 0
                     and (metadata[i]["technical_update_eligible"] or not only_safe)]
            if not legal:
                plans[key][task] = []
                continue
            legal.sort(key=lambda i: (metadata[i]["full_tokens"], metadata[i]["dataset_local_index"]))
            picked = []
            for label, quantile in [("short_p10", .10), ("middle_p50", .50), ("long_p90", .90), ("extreme_longest_legal", 1.)]:
                i = legal[int((len(legal) - 1) * quantile)]
                sample_id = metadata[i]["sample_id"]
                picked.append(sample_id)
                reasons[sample_id].append(key + ":" + label)
            plans[key][task] = picked
    allocation = largest_remainder([ranges[t][1] - ranges[t][0] for t in TASKS], 128)
    for group in range(2):
        selected = []
        for task, count in zip(TASKS, allocation):
            start, end = ranges[task]
            eligible = [i for i in range(start, end) if metadata[i]["technical_update_eligible"]]
            if len(eligible) < count:
                raise RuntimeError("Insufficient verified training-period diagnostic records for " + task)
            picked = rng.choice(eligible, count, replace=False).tolist()
            selected += [metadata[i]["sample_id"] for i in picked]
        rng.shuffle(selected)
        plans["mixed_global128_" + str(group)] = selected
        for sample_id in selected:
            reasons[sample_id].append("mixed_global128_" + str(group))
    # The longest and shortest legal rows of each task deliberately share
    # microbatches. This plan is diagnostic and is never an update plan.
    stress = []
    for task in TASKS:
        start, end = ranges[task]
        legal = [i for i in range(start, end) if not metadata[i]["full_truncated"]
                 and metadata[i]["shift_effective_label_tokens"] > 0]
        legal.sort(key=lambda i: (metadata[i]["full_tokens"], i))
        for i in [legal[0], legal[-1], legal[1], legal[-2]]:
            sample_id = metadata[i]["sample_id"]
            stress.append(sample_id)
            reasons[sample_id].append("padding_stress_diagnostic_only")
    plans["padding_stress"] = stress
    return plans, reasons, allocation


def run(cli, report):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(cli.original))
    import numpy as np
    import torch
    import transformers

    torch.set_num_threads(cli.cpu_threads)
    official = {name: importlib.import_module(name) for name in ["utils", "data", "prompt", "collator"]}
    for name, module in official.items():
        assert Path(module.__file__).resolve().parent == cli.original, (name, module.__file__)
    report["environment"] = {"python_executable": sys.executable, "python_version": sys.version,
                             "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
                             "cuda_visible_devices": "", "model_instantiated": False,
                             "torch_cpu_threads": cli.cpu_threads}
    report["source_files"] = {name: {"path": str(cli.original / name), "sha256": file_sha256(cli.original / name)}
                              for name in ["data.py", "utils.py", "collator.py", "prompt.py", "lora_finetune.py", "run.sh"]}
    parser = argparse.ArgumentParser(add_help=False)
    for fn in [official["utils"].parse_global_args, official["utils"].parse_train_args, official["utils"].parse_dataset_args]:
        parser = fn(parser)
    args = parser.parse_args([])
    for key, value in {"seed": 42, "data_path": str(cli.data_root), "dataset": "Instruments", "base_model": str(cli.model),
                       "model_max_length": 2048, "max_his_len": 20, "index_file": ".index.json", "only_train_response": True,
                       "tasks": ",".join(TASKS), "train_prompt_sample_num": "1,1,1,1,1,1",
                       "train_data_sample_num": "0,0,0,20000,0,0"}.items():
        setattr(args, key, value)
    report["args_namespace"] = vars(args)
    data_dir = cli.data_root / "Instruments"
    loaded = {}
    report["assets"] = {}
    for name in ["Instruments.inter.json", "Instruments.item.json", "Instruments.index.json", "Instruments.user.json"]:
        path = data_dir / name
        loaded[name] = json.loads(path.read_text())
        report["assets"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    assert file_sha256(cli.model / "tokenizer.model") == EXPECTED_TOKENIZER_SHA256
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    rng_states = {"seed": 42, "before_dataset_loader": rng_snapshot(np, torch)}
    started = time.monotonic()
    train, valid = official["utils"].load_datasets(args)
    datasets = dict(zip(TASKS, train.datasets))
    actual_rows = [len(datasets[task]) for task in TASKS]
    assert actual_rows == EXPECTED_ROWS, (actual_rows, EXPECTED_ROWS)
    assert len(train) == 221225
    report["dataset_loader_seconds"] = time.monotonic() - started
    report["dataset_rows"] = dict(zip(TASKS, actual_rows))
    report["concat_rows"] = len(train)
    report["validation_protocol"] = {"users": len(valid.inter_data), "actual_constructed_records": len(valid),
                                     "sample_valid": args.sample_valid, "prompt_sample_num": args.valid_prompt_sample_num,
                                     "valid_prompt_id_if_not_sampled": args.valid_prompt_id,
                                     "per_epoch_nominal_records": len(valid), "validation_scanned": False,
                                     "note": "The original loader constructs two sampled templates per user; no validation loss/model scan here."}
    rng_states["after_dataset_loader"] = rng_snapshot(np, torch)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(str(cli.model), local_files_only=True,
                model_max_length=2048, padding_side="right")
    tokenizer.pad_token_id = 0
    assert len(tokenizer) == 32000
    new_tokens = datasets["seqrec"].get_new_tokens()
    assert len(new_tokens) == 930
    assert tokenizer.add_tokens(new_tokens) == 930
    mappings = [{"token": token, "id": tokenizer.convert_tokens_to_ids(token)} for token in new_tokens]
    assert [row["id"] for row in mappings] == list(range(32000, 32930))
    assert all(tokenizer.encode(row["token"], add_special_tokens=False) == [row["id"]] for row in mappings)
    json_dump(cli.output / "sid_token_map.json", mappings)
    report["tokenizer"] = {"path": str(cli.model), "class": type(tokenizer).__name__, "base_tokens": 32000,
                            "added_tokens": 930, "vocabulary_size": len(tokenizer), "padding_side": tokenizer.padding_side,
                            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                            "pad_token_id": tokenizer.pad_token_id, "unk_token_id": tokenizer.unk_token_id,
                            "add_bos_token": tokenizer.add_bos_token, "add_eos_token": tokenizer.add_eos_token,
                            "model_max_length": tokenizer.model_max_length,
                            "tokenizer_model_sha256": EXPECTED_TOKENIZER_SHA256}
    collator = official["collator"].Collator(args, tokenizer)
    origins = Origins(loaded, datasets, args, np)
    report["auxiliary_data_status"] = origins.summary()
    observed = ObservedRandom(random)
    official["data"].random = observed
    metadata, ranges, task_rng_states, task_times = [], {}, {}, {}
    with gzip.open(cli.output / "records.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as stream:
        for task in TASKS:
            task_start = time.monotonic()
            ds = datasets[task]
            task_rng_states[task] = random.getstate()
            ranges[task] = [len(metadata), len(metadata) + len(ds)]
            print(json.dumps({"event": "task_start", "task": task, "rows": len(ds)}), flush=True)
            for start in range(0, len(ds), cli.chunk_size):
                rows, observations, provenance = [], [], []
                for local in range(start, min(start + cli.chunk_size, len(ds))):
                    origin, safe = origins.at(task, local)
                    observed.reset()
                    row = ds[local]
                    assert row["labels"].startswith(row["input_ids"])
                    rows.append(row)
                    observations.append(observed.events[:])
                    provenance.append((origin, safe))
                # Actual original collator, including its tokenizer(text_target)
                # behavior. Raw lengths are separately measured without padding
                # or truncation. No custom mask replaces the original mask.
                batch = collator(rows)
                raw_fulls = tokenizer([row["labels"] + tokenizer.eos_token for row in rows],
                                     padding=False, truncation=False)["input_ids"]
                raw_inputs = tokenizer([row["input_ids"] for row in rows], padding=False, truncation=False)["input_ids"]
                answers = tokenizer([row["labels"][len(row["input_ids"]):] for row in rows],
                                    padding=False, truncation=False, add_special_tokens=False)["input_ids"]
                for pos, row in enumerate(rows):
                    local = start + pos
                    origin, safe = provenance[pos]
                    metrics = row_metrics(row, batch, pos, tokenizer, raw_fulls[pos], raw_inputs[pos], answers[pos], torch)
                    events = observations[pos]
                    prompt_id = next(event["value"] for event in events if event["kind"] == "randint")
                    safe = bool(safe and metrics["shift_effective_label_tokens"] > 0)
                    record = {"sample_id": task + ":" + str(local), "task": task, "dataset_local_index": local,
                              "concat_index": len(metadata), "prompt_id": prompt_id, "random_choices": events,
                              "technical_update_eligible": safe, "diagnostic_only": not safe,
                              "origin_user_id": origin.get("user_id"),
                              "origin_item_id": origin.get("target_item_id", origin.get("item_id")),
                              "training_period_status": origin["training_period_status"], **metrics}
                    metadata.append(record)
                    stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                if start % (cli.chunk_size * 100) == 0:
                    print(json.dumps({"event": "task_progress", "task": task, "rows_done": min(start + cli.chunk_size, len(ds))}), flush=True)
            task_times[task] = time.monotonic() - task_start
            print(json.dumps({"event": "task_done", "task": task, "seconds": task_times[task]}), flush=True)
    rng_states["task_getitem_start_states"] = task_rng_states
    rng_states["after_full_sequential_getitem_pass"] = rng_snapshot(np, torch)
    report["realization"] = {"seed": 42, "loader": "unmodified official utils.load_datasets",
                             "full_getitem_passes_for_statistics": 1,
                             "order": "fixed task order then ascending original Dataset local index",
                             "original_random_calls_observed_without_changing_rng_consumption": True,
                             "all_possible_prompt_variants_enumerated": False,
                             "boundary": "One deterministic realization of stochastic prompt/preference/query selection, not an expectation over all variants or all epochs.",
                             "task_seconds": task_times, "analysis_chunk_size": cli.chunk_size,
                             "analysis_chunk_padding_is_not_training_microbatch_padding": True}
    report["per_task"] = {task: describe_subset(metadata, range(*ranges[task]), np) for task in TASKS}
    report["joint"] = describe_subset(metadata, range(len(metadata)), np)
    # Shuffle only for CPU padding diagnostics. This does not change the
    # official dataset, the distribution pass, or the future Trainer sampler.
    shuffled = np.random.RandomState(42).permutation(len(metadata)).tolist()
    joint_lengths = [metadata[i]["full_tokens"] for i in shuffled]
    report["joint"]["random_mixed_padding_diagnostic"] = {str(micro): padding_summary(joint_lengths, micro, np)
                                                            for micro in [1, 2, 4, 8]}
    plans, reasons, allocation = select_plans(metadata, ranges, np)
    by_sample = {row["sample_id"]: row for row in metadata}
    records = {}
    for task in TASKS:
        wanted = {by_sample[sample_id]["dataset_local_index"] for sample_id in reasons if by_sample[sample_id]["task"] == task}
        if not wanted:
            continue
        random.setstate(task_rng_states[task])
        for local in range(max(wanted) + 1):
            if local in wanted:
                before = random.getstate()
            observed.reset()
            row = datasets[task][local]
            if local not in wanted:
                continue
            sample_id = task + ":" + str(local)
            meta = by_sample[sample_id]
            assert text_sha256(row["input_ids"]) == meta["input_sha256_utf8"]
            assert text_sha256(row["labels"]) == meta["full_sha256_utf8"]
            single = collator([row])
            origin, safe = origins.at(task, local)
            records[sample_id] = {"sample_id": sample_id, "task": task, "dataset_local_index": local,
                                  "dataset_index": local, "concat_index": meta["concat_index"],
                                  "prompt_id": meta["prompt_id"], "python_random_state_before_getitem": before,
                                  "random_choices": observed.events[:], "input_ids": row["input_ids"], "labels": row["labels"],
                                  "origin": origin, "technical_update_eligible": meta["technical_update_eligible"],
                                  "diagnostic_only": meta["diagnostic_only"], "selection_reasons": reasons[sample_id],
                                  "collated": {key: value[0].tolist() for key, value in single.items()}, "metrics": meta}
    samples = {"schema_version": 1, "seed": 42, "args_namespace": vars(args), "records": records, "plans": plans,
               "mixed_global128_task_counts": dict(zip(TASKS, allocation)),
               "selection_algorithm": {"representatives": "stable sort by actual one-realization full token length; floor((n-1)*q), q=.10,.50,.90,1; exclude truncation and no shifted supervision for legal probes",
                                       "mixed": "largest-remainder task counts for global128; seed42 NumPy RandomState; uniform without-replacement per-task technical-eligible diagnostic subset, then shuffle; two plans consume same RNG stream",
                                       "mixed_limit": "Task proportions match the original full dataset to integer rounding. Auxiliary rows are restricted for safe technical updates; their within-task length distribution is not claimed identical to all mismatched rows.",
                                       "padding_stress": "per task shortest, longest, second shortest, second longest legal realization; diagnostic only",
                                       "sampling_never_uses_loss_or_recommendation_metrics": True},
               "replay_instruction": "Recursive-convert python_random_state_before_getitem lists to tuples; random.setstate(state); original_dataset[dataset_local_index] must exactly equal input_ids/labels. Rebuild original datasets after seed42 in original task order first.",
               "data_boundary": origins.summary()}
    json_dump(cli.output / "samples.json", samples)
    json_dump(cli.output / "rng_states.json", rng_states)
    report["performance_samples"] = {"file": "samples.json", "distinct_records": len(records),
                                      "mixed_global128_task_counts": dict(zip(TASKS, allocation)),
                                      "plan_sizes": {key: len(value) for key, value in plans.items() if isinstance(value, list)},
                                      "all_selected_records_replayed_and_text_hash_checked": True,
                                      "formal_sampler_changed": False}
    # A short observed validation sample permits GPU-side timing without a
    # full validation scan. Official construction sampled two prompts/user.
    valid_order = sorted(range(len(valid.inter_data)), key=lambda i: (len(valid.inter_data[i]["inters"]), i))
    selected_valid = [valid_order[int((len(valid_order) - 1) * q)] for q in [.1, .5, .9]]
    valid_samples = []
    for user_pos in selected_valid:
        uid = list(loaded["Instruments.inter.json"])[user_pos]
        for template_slot in range(args.valid_prompt_sample_num):
            index = user_pos * args.valid_prompt_sample_num + template_slot
            row = valid[index]
            batch = collator([row])
            valid_samples.append({"validation_index": index, "user_id": uid, "template_slot": template_slot,
                                  "input_ids": row["input_ids"], "labels": row["labels"],
                                  "collated": {key: value[0].tolist() for key, value in batch.items()}})
    json_dump(cli.output / "validation_samples.json", {"args_namespace": vars(args), "records": valid_samples,
              "selection": "Original sampled validation templates; users at p10/p50/p90 of history-string length, fixed index tie break; bounded six records, no validation metrics."})
    assert not torch.cuda.is_initialized(), "Unexpected CUDA context in CPU analysis"
    report["cuda_context_initialized"] = False
    report["status"] = "PASS_WITH_FORMAL_DATA_BLOCKER"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    cli = parser.parse_args()
    if not 1 <= cli.chunk_size <= 512 or not 1 <= cli.cpu_threads <= 16:
        parser.error("chunk-size must be 1..512 and cpu-threads 1..16")
    for key in ["original", "data_root", "model", "output"]:
        setattr(cli, key, getattr(cli, key).resolve())
    cli.output.mkdir(parents=True, exist_ok=True)
    if (cli.output / "workload_summary.json").exists():
        parser.error("Refusing to overwrite an existing workload_summary.json; use a distinct attempt output directory")
    started = time.monotonic()
    report = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "argv": sys.argv,
              "cwd": os.getcwd(), "pid": os.getpid(), "scope": "CPU actual official six-task Dataset/tokenizer/collator; no model/GPU",
              "status": "RUNNING"}
    rc = 0
    try:
        run(cli, report)
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        rc = 1
    report["wall_seconds"] = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["resources"] = {"peak_rss_bytes": int(usage.ru_maxrss * 1024), "cpu_user_seconds": usage.ru_utime,
                           "cpu_system_seconds": usage.ru_stime, "scope": "This Linux CPU audit process only"}
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["exit_code"] = rc
    json_dump(cli.output / "workload_summary.json", report)
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in cli.output.iterdir()
             if path.is_file() and path.name != "output_manifest.json"}
    json_dump(cli.output / "output_manifest.json", files)
    print(json.dumps({"event": "finished", "status": report["status"], "exit_code": rc,
                      "wall_seconds": report["wall_seconds"]}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
