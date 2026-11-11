"""CPU replay of the audited LC-Rec realization using the two Qwen tokenizers."""

from __future__ import annotations

import argparse
import collections
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
from types import SimpleNamespace

from common import (Allowed, Collator, EXPECTED_ROWS, MODELS, TARGETS, TASKS,
                    model_class_and_config, sha, text_sha, tokenizer_for, write)


class RecordedRandom:
    def __init__(self):
        self.events = []
        self.position = 0

    def bind(self, events):
        self.events = events
        self.position = 0

    def take(self, kind):
        event = self.events[self.position]
        self.position += 1
        assert event["kind"] == kind, (event, kind)
        return event

    def randint(self, lower, upper):
        event = self.take("randint")
        assert (event["a"], event["b"]) == (lower, upper)
        assert lower <= event["value"] <= upper
        return event["value"]

    def choice(self, values):
        event = self.take("choice")
        assert event["population"] == len(values)
        positions = event["matching_value_indices"]
        assert positions and all(values[position] == values[positions[0]] for position in positions)
        return values[positions[0]]

    def finished(self):
        assert self.position == len(self.events), (self.position, len(self.events))


def summary(values, numpy):
    array = numpy.asarray(values, dtype=numpy.float64)
    if not len(array):
        return {"count": 0}
    quantiles = numpy.percentile(array, [10, 50, 90, 99], method="linear")
    return {
        "count": len(array), "min": float(array.min()), "p10": float(quantiles[0]),
        "p50": float(quantiles[1]), "p90": float(quantiles[2]), "p99": float(quantiles[3]),
        "max": float(array.max()), "mean": float(array.mean()), "sum": int(array.sum()),
    }


def padding(lengths, micro, numpy):
    array = numpy.asarray(lengths, dtype=numpy.int64)
    useful = int(array.sum())
    slots = sum(int(array[begin:begin + micro].max()) * len(array[begin:begin + micro])
                for begin in range(0, len(array), micro))
    return {
        "micro": micro, "records": len(array), "useful_tokens": useful,
        "allocated_token_slots": slots, "padding_tokens": slots - useful,
        "padding_ratio": 1 - useful / slots if slots else 0,
    }


def replay(dataset, metadata, observer):
    observer.bind(metadata["random_choices"])
    row = dataset[metadata["dataset_local_index"]]
    observer.finished()
    assert text_sha(row["input_ids"]) == metadata["input_sha256_utf8"], metadata["sample_id"]
    assert text_sha(row["labels"]) == metadata["full_sha256_utf8"], metadata["sample_id"]
    return row


def record_metrics(row, raw_prompt, raw_full, answer, batch, position, tokenizer):
    labels = batch["labels"][position]
    active = labels != -100
    supervised = labels[active].tolist()
    useful = int(batch["attention_mask"][position].sum())
    prompt_count = min(len(raw_prompt), tokenizer.model_max_length)
    return {
        "input_raw_tokens": len(raw_prompt), "answer_standalone_tokens": len(answer),
        "full_raw_tokens_with_eos": len(raw_full), "input_tokens": prompt_count,
        "full_tokens": useful, "effective_label_tokens": int(active.sum()),
        "shift_effective_label_tokens": int((labels[1:] != -100).sum()),
        "raw_response_suffix_tokens_with_eos": max(0, len(raw_full) - len(raw_prompt)),
        "input_truncated": len(raw_prompt) > tokenizer.model_max_length,
        "full_truncated": len(raw_full) > tokenizer.model_max_length,
        "answer_text_empty": row["labels"] == row["input_ids"],
        "zero_supervision": not supervised,
        "eos_only_supervision": bool(supervised) and all(token == tokenizer.eos_token_id for token in supervised),
        "supervised_eos_tokens": sum(token == tokenizer.eos_token_id for token in supervised),
        "prompt_full_token_prefix_match": raw_full[:prompt_count] == raw_prompt[:prompt_count],
        "ends_with_eos_after_truncation": bool(useful) and raw_full[useful - 1] == tokenizer.eos_token_id,
        "starts_with_bos": bool(raw_full) and raw_full[0] == tokenizer.bos_token_id,
    }


def describe(metadata, positions, numpy):
    selected = [metadata[position] for position in positions]
    numeric = ["input_raw_tokens", "answer_standalone_tokens", "full_raw_tokens_with_eos",
               "input_tokens", "full_tokens", "effective_label_tokens", "shift_effective_label_tokens"]
    flags = ["input_truncated", "full_truncated", "answer_text_empty", "zero_supervision",
             "eos_only_supervision", "prompt_full_token_prefix_match", "ends_with_eos_after_truncation",
             "technical_update_eligible", "diagnostic_only"]
    return {
        **{field: summary([entry[field] for entry in selected], numpy) for field in numeric},
        "counts": {field: sum(bool(entry[field]) for entry in selected) for field in flags},
        "prompt_id_counts": dict(collections.Counter(entry["prompt_id"] for entry in selected)),
        "padding_task_index_order": {
            str(micro): padding([entry["full_tokens"] for entry in selected], micro, numpy)
            for micro in [1, 2, 4, 8]
        },
    }


def make_plans(metadata, old_samples):
    by_id = {entry["sample_id"]: entry for entry in metadata}
    plans = {name: list(old_samples["plans"][name]) for name in ["mixed_global128_0", "mixed_global128_1"]}
    plans["task_representatives"] = {}
    plans["task_technical_representatives"] = {}
    for task in TASKS:
        for name, require_safe in [("task_representatives", False), ("task_technical_representatives", True)]:
            legal = [entry for entry in metadata if entry["task"] == task
                     and not entry["full_truncated"] and entry["shift_effective_label_tokens"] > 0
                     and (entry["technical_update_eligible"] or not require_safe)]
            legal.sort(key=lambda entry: (entry["full_tokens"], entry["dataset_local_index"]))
            assert legal, (task, name)
            plans[name][task] = [legal[int((len(legal) - 1) * quantile)]["sample_id"]
                                 for quantile in [.1, .5, .9, 1.]]
    tail = plans["mixed_global128_1"][:]
    for task in TASKS:
        longest = plans["task_technical_representatives"][task][-1]
        if longest not in tail:
            replacement = next(position for position, sample_id in enumerate(tail) if by_id[sample_id]["task"] == task)
            tail[replacement] = longest
    plans["mixed_global128_tail"] = tail
    for name in ["mixed_global128_0", "mixed_global128_1", "mixed_global128_tail"]:
        assert len(plans[name]) == 128 and len(set(plans[name])) == 128
        assert all(by_id[sample_id]["technical_update_eligible"] for sample_id in plans[name])
        assert all(by_id[sample_id]["shift_effective_label_tokens"] > 0 for sample_id in plans[name])
        assert dict(collections.Counter(by_id[sample_id]["task"] for sample_id in plans[name])) == old_samples["mixed_global128_task_counts"]
    return plans


def selected_origin(task, local_index, datasets, old_metadata):
    origin = {
        "training_period_status": old_metadata["training_period_status"],
        "old_cpu_sample_id": old_metadata["sample_id"],
        "verified_unchanged_realization_sha256": True,
    }
    if task not in ["itemsearch", "preferenceobtain"]:
        return origin
    user_info = datasets[task].user_info
    user_id = list(user_info["user_explicit_preference"])[local_index]
    sequence = list(map(str, datasets["seqrec"].inters[user_id]))
    auxiliary = user_info["user_vague_intention"]["train"][user_id]
    paired = (str(auxiliary["item"]) == sequence[-3]
              and list(map(str, auxiliary["inters"])) == sequence[:-3])
    if old_metadata["technical_update_eligible"]:
        assert paired, (task, local_index)
    origin.update({
        "user_ref": text_sha("LC-Rec Instruments user:" + user_id),
        "current_selected_auxiliary_pairing_matches_training_split": paired,
        "scope": "Selected records only; inherited complete historical audit is not rerun",
        "explicit_preference_provenance_limit": "Per-record generation context remains unproven; technical selection does not certify clean formal training data",
    })
    return origin


def meta_inventory(model_path, tokenizer, report):
    from accelerate import init_empty_weights
    from peft import LoraConfig, TaskType, get_peft_model

    model_class, model_config = model_class_and_config(model_path)
    original_rows = model_config.vocab_size
    with init_empty_weights():
        model = model_class(model_config)
    model.tie_weights()
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.tie_weights()
    full_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    report["model"] = {
        "class": model_class.__name__, "base_vocab_rows": original_rows,
        "expanded_vocab_rows": len(tokenizer), "physical_rows_added": len(tokenizer) - original_rows,
        "original_unique_parameters": base_parameters, "expanded_unique_parameters": full_parameters,
        "tied_embedding_head": True, "architecture_modified": False,
        "new_row_initialization": "Original LC-Rec Transformers-era normal initializer; mean_resizing=False; IDs may reuse base padded physical vocabulary rows",
        "inventory": {name: {"shape": list(parameter.shape), "numel": parameter.numel()}
                      for name, parameter in model.named_parameters()},
    }
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=.05, target_modules=TARGETS,
        modules_to_save=["embed_tokens", "lm_head"], ensure_weight_tying=True,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    physical = sum(parameter.numel() for parameter in model.parameters())
    report["lora_static"] = {
        "rank": 8, "alpha": 32, "dropout": .05, "targets": TARGETS,
        "modules_to_save": ["embed_tokens", "lm_head"], "ensure_weight_tying": True,
        "lora_parameters": sum(parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name),
        "trainable_parameters": trainable, "physical_unique_parameters": physical,
        "meta_only": True,
    }
    report["memory_static"] = {
        "scope": "Estimates, not GPU measurements; two equal ZeRO-2 partitions",
        "full_bf16_weights_bytes_per_gpu": 2 * full_parameters,
        "full_zero2_persistent_bytes_per_gpu": 9 * full_parameters,
        "lora_bf16_weights_bytes_per_gpu": 2 * physical,
        "lora_zero2_persistent_bytes_per_gpu": 2 * physical + 7 * trainable,
        "logits_float32_bytes_per_micro_and_token": 4 * len(tokenizer),
        "excludes": ["activations", "communication buckets", "allocator reserve", "temporary float32 CE tensors"],
    }
    report["checkpoint_static"] = {
        "full_bf16_model_bytes": 2 * full_parameters,
        "full_fp32_master_and_adam_bytes_global": 12 * full_parameters,
        "full_training_state_lower_bound_bytes": 14 * full_parameters,
        "lora_adapter_trainable_bf16_bytes": 2 * trainable,
        "lora_fp32_master_and_adam_bytes_global": 12 * trainable,
        "scheduler_rng_metadata_bytes": "NOT MEASURED",
        "full_training_state_restore": "NOT TESTED by CPU audit",
    }


def run(arguments, report):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU audit must hide CUDA devices before importing torch"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy
    import torch
    import transformers

    config = json.loads(arguments.config.read_text())
    realization = Path(config["cpu_realization_dir"])
    old_summary = json.loads((realization / "workload_summary.json").read_text())
    old_samples = json.loads((realization / "samples.json").read_text())
    old_validation = json.loads((realization / "validation_samples.json").read_text())
    sys.path.insert(0, str(arguments.original))
    modules = {name: importlib.import_module(name) for name in ["data", "utils", "prompt", "collator"]}
    assert all(Path(module.__file__).resolve().parent == arguments.original for module in modules.values())
    report["run_commit"] = config.get("run_commit", os.environ.get("RUN_COMMIT"))
    report["model_identity"] = {"key": arguments.model, "name": MODELS[arguments.model][0], "revision": MODELS[arguments.model][1], "path": str(arguments.model_path)}
    assert arguments.model_path.name == MODELS[arguments.model][1]
    report["source_files"] = {}
    for filename in ["data.py", "collator.py", "utils.py", "prompt.py", "lora_finetune.py", "finetune.py", "run.sh", "test.py", "test_ddp.py"]:
        source = arguments.original / filename
        digest = sha(source)
        if filename in old_summary["source_files"]:
            assert digest == old_summary["source_files"][filename]["sha256"]
        report["source_files"][filename] = {"path": str(source), "sha256": digest}
    data_directory = arguments.data if arguments.data.name == "Instruments" else arguments.data / "Instruments"
    report["assets"] = {}
    for filename, prior in old_summary["assets"].items():
        current = data_directory / filename
        digest = sha(current)
        assert digest == prior["sha256"], filename
        report["assets"][filename] = {"path": str(current), "bytes": current.stat().st_size, "sha256": digest}
    report["realization"] = {
        "source_directory": str(realization),
        "old_files": {name: {"sha256": sha(realization / name), "bytes": (realization / name).stat().st_size}
                      for name in ["records.jsonl.gz", "samples.json", "validation_samples.json", "workload_summary.json"]},
        "rule": "Replay original recorded randint/choice outcomes for all 221225 unchanged Dataset rows and assert both original natural-language text hashes",
        "shared_across_qwen_models": True,
        "new_prompt_sampling": False,
        "scope": "One audited seed42 realization, not enumeration of stochastic prompts across epochs",
    }
    dataset_args = SimpleNamespace(**old_samples["args_namespace"])
    dataset_args.data_path = str(data_directory.parent)
    dataset_args.base_model = str(arguments.model_path)
    dataset_args.model_max_length = 2048
    assert dataset_args.tasks == ",".join(TASKS)
    assert dataset_args.train_prompt_sample_num == "1,1,1,1,1,1"
    assert dataset_args.train_data_sample_num == "0,0,0,20000,0,0"
    assert dataset_args.max_his_len == 20 and dataset_args.only_train_response
    random.seed(42)
    numpy.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    train, validation = modules["utils"].load_datasets(dataset_args)
    datasets = dict(zip(TASKS, train.datasets))
    assert [len(datasets[task]) for task in TASKS] == EXPECTED_ROWS
    assert len(train) == 221225 and len(validation) == 49544
    report["args_namespace"] = vars(dataset_args)
    report["dataset_rows"] = dict(zip(TASKS, EXPECTED_ROWS))
    report["concat_rows"] = len(train)
    report["auxiliary_data_status"] = old_summary["auxiliary_data_status"]
    report["validation_protocol"] = old_summary["validation_protocol"]
    report["validation_protocol"]["current_constructed_rows"] = len(validation)
    report["environment"] = {
        "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": numpy.__version__,
        "cpu_threads": torch.get_num_threads(), "cuda_visible_devices": "", "model_payload_loaded": False,
    }
    tokens = datasets["seqrec"].get_new_tokens()
    tokenizer, report["tokenizer"] = tokenizer_for(arguments.model_path, tokens)
    report["tokenizer_source_files"] = {
        name: {"sha256": sha(arguments.model_path / name), "bytes": (arguments.model_path / name).stat().st_size}
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]
        if (arguments.model_path / name).exists()
    }
    indices = datasets["seqrec"].indices
    assert len(indices) == 9922 and len({"".join(parts) for parts in indices.values()}) == 9922
    tokenizer.save_pretrained(arguments.out / "tokenizer")
    restored = transformers.AutoTokenizer.from_pretrained(arguments.out / "tokenizer", local_files_only=True)
    assert restored.get_vocab() == tokenizer.get_vocab()
    for candidate in [tokenizer, restored]:
        for parts in indices.values():
            assert len(parts) == 4
            expected = [candidate.convert_tokens_to_ids(token) for token in parts]
            encoded = candidate.encode("".join(parts), add_special_tokens=False)
            assert encoded == expected and len(encoded) == 4
            assert candidate.decode(encoded, skip_special_tokens=False) == "".join(parts)
    report["tokenizer"].update({"save_reload_vocab_exact": True, "all_9922_item_sids_four_tokens_exact_before_after_reload": True})
    write(arguments.out / "sid_token_map.json", [{"token": token, "id": tokenizer.convert_tokens_to_ids(token)} for token in tokens])
    collator = Collator(tokenizer)
    literal_collator = modules["collator"].Collator(dataset_args, tokenizer)
    observer = RecordedRandom()
    modules["data"].random = observer
    metadata = []
    by_task_positions = {task: [] for task in TASKS}
    boundary_examples = []
    literal_eos_difference_positions = 0
    chunk_size = int(config.get("cpu_chunk_size", 256))
    with gzip.open(realization / "records.jsonl.gz", "rt", encoding="utf-8") as stream:
        while True:
            old_chunk = []
            for unused in range(chunk_size):
                line = stream.readline()
                if not line:
                    break
                old_chunk.append(json.loads(line))
            if not old_chunk:
                break
            rows = [replay(datasets[entry["task"]], entry, observer) for entry in old_chunk]
            batch = collator(rows)
            literal = literal_collator(rows)
            assert torch.equal(batch["input_ids"], literal["input_ids"])
            assert torch.equal(batch["attention_mask"], literal["attention_mask"])
            different = batch["labels"] != literal["labels"]
            if bool(different.any()):
                assert tokenizer.pad_token_id == tokenizer.eos_token_id
                assert bool(torch.all(batch["labels"][different] == tokenizer.eos_token_id))
                assert bool(torch.all(literal["labels"][different] == -100))
                literal_eos_difference_positions += int(different.sum())
            prompts = tokenizer([row["input_ids"] for row in rows], padding=False, truncation=False)["input_ids"]
            fulls = tokenizer([row["labels"] + tokenizer.eos_token for row in rows], padding=False, truncation=False)["input_ids"]
            answers = tokenizer([row["labels"][len(row["input_ids"]):] for row in rows], padding=False,
                                truncation=False, add_special_tokens=False)["input_ids"]
            for position, (entry, row) in enumerate(zip(old_chunk, rows)):
                metrics = record_metrics(row, prompts[position], fulls[position], answers[position], batch, position, tokenizer)
                current = {key: entry[key] for key in ["sample_id", "task", "dataset_local_index", "concat_index", "prompt_id", "random_choices",
                                                       "technical_update_eligible", "diagnostic_only", "training_period_status", "input_sha256_utf8", "full_sha256_utf8"]}
                current.update(metrics)
                assert current["concat_index"] == len(metadata)
                by_task_positions[current["task"]].append(len(metadata))
                metadata.append(current)
                if not metrics["prompt_full_token_prefix_match"] and len(boundary_examples) < 16:
                    offsets = tokenizer(row["labels"] + tokenizer.eos_token, return_offsets_mapping=True)["offset_mapping"]
                    boundary_examples.append({
                        "sample_id": entry["sample_id"], "prompt_chars": len(row["input_ids"]),
                        "separately_encoded_prompt_tokens": len(prompts[position]),
                        "full_boundary_tokens": fulls[position][max(0, len(prompts[position]) - 3):len(prompts[position]) + 3],
                        "full_boundary_offsets": offsets[max(0, len(prompts[position]) - 3):len(prompts[position]) + 3],
                        "answer_prefix": row["labels"][len(row["input_ids"]):len(row["input_ids"]) + 64],
                    })
            if len(metadata) % (chunk_size * 100) == 0 or len(metadata) == len(train):
                print(json.dumps({"event": "qwen_length_progress", "model": arguments.model, "rows": len(metadata), "total": len(train)}), flush=True)
    assert len(metadata) == len(train)
    assert [len(by_task_positions[task]) for task in TASKS] == EXPECTED_ROWS
    report["per_task"] = {task: describe(metadata, by_task_positions[task], numpy) for task in TASKS}
    report["joint"] = describe(metadata, range(len(metadata)), numpy)
    shuffled = numpy.random.RandomState(42).permutation(len(metadata))
    report["joint"]["random_mixed_padding_diagnostic"] = {
        str(micro): padding([metadata[int(position)]["full_tokens"] for position in shuffled], micro, numpy)
        for micro in [1, 2, 4, 8]
    }
    report["mask"] = {
        "checked_records": len(metadata), "original_natural_language_text_hashes_exact": True,
        "literal_official_collator_ids_and_attention_exact": True,
        "literal_official_collator_difference_positions": literal_eos_difference_positions,
        "difference": "Only real EOS supervision restored where Qwen PAD=EOS makes original token-ID-based padding mask erase EOS",
        "prompt_mask": "Original independently encoded prompt length; no answer-only four-SID assumption for text-output tasks",
        "boundary_mismatch_count": sum(not entry["prompt_full_token_prefix_match"] for entry in metadata),
        "boundary_examples": boundary_examples,
        "boundary_policy": "Preserve original LC-Rec prompt-length mask, including tokenizer boundary behavior; no offset-based supervision rewrite",
        "causal_shift": True, "cross_entropy_temperature": 1.0,
        "padding_labels_all_ignored": True, "automatic_chat_template": False,
    }
    plans = make_plans(metadata, old_samples)
    by_id = {entry["sample_id"]: entry for entry in metadata}
    wanted = set()
    for name, values in plans.items():
        if isinstance(values, dict):
            wanted.update(sample_id for task_values in values.values() for sample_id in task_values)
        else:
            wanted.update(values)
    records = {}
    for sample_id in sorted(wanted, key=lambda value: by_id[value]["concat_index"]):
        entry = by_id[sample_id]
        row = replay(datasets[entry["task"]], entry, observer)
        batch = collator([row])
        records[sample_id] = {
            **{key: entry[key] for key in ["sample_id", "task", "dataset_local_index", "concat_index", "prompt_id", "technical_update_eligible", "diagnostic_only"]},
            **row, "metrics": {key: value for key, value in entry.items() if key not in ["random_choices"]},
            "collated": {key: value[0].tolist() for key, value in batch.items()},
            "origin": old_samples["records"].get(sample_id, {}).get("origin", selected_origin(entry["task"], entry["dataset_local_index"], datasets, entry)),
        }
    valid_records = []
    for prior in old_validation["records"]:
        row = validation[prior["validation_index"]]
        assert row["input_ids"] == prior["input_ids"] and row["labels"] == prior["labels"]
        batch = collator([row])
        valid_records.append({
            "validation_index": prior["validation_index"], "template_slot": prior["template_slot"],
            "user_ref": text_sha("LC-Rec Instruments user:" + str(prior["user_id"])),
            **row, "collated": {key: value[0].tolist() for key, value in batch.items()},
        })
    test_dataset = modules["data"].SeqRecDataset(dataset_args, mode="test", prompt_id=0, sample_num=-1)
    assert len(test_dataset) == 24772 and len(test_dataset.prompts) == 17
    user_ids = list(test_dataset.inters)
    test_order = sorted(range(len(test_dataset)), key=lambda position: (len(test_dataset.inter_data[position]["inters"]), position))
    generation = []
    callback = Allowed(tokenizer, indices)
    delimiter_checks = []
    for bucket, quantile in [("short_p10", .1), ("middle_p50", .5), ("boundary_longest", 1.)]:
        row_index = test_order[int((len(test_order) - 1) * quantile)]
        for prompt_id in range(17):
            test_dataset.set_prompt(prompt_id)
            row = test_dataset[row_index]
            input_ids = tokenizer(row["input_ids"], truncation=True, max_length=2048)["input_ids"]
            callback.bind(input_ids, row["input_ids"])
            current_ids = input_ids[:]
            assert set(callback(0, torch.tensor(current_ids))) == callback.allowed[0]
            for position in range(4):
                current_ids.append(min(callback.allowed[position]))
                assert set(callback(0, torch.tensor(current_ids))) == callback.allowed[position + 1]
            generation.append({
                "row_idx": row_index, "user_ref": text_sha("LC-Rec Instruments user:" + user_ids[row_index]),
                "history_bucket": bucket, "prompt_id": prompt_id, "input_text": row["input_ids"],
                "target_sid": row["labels"], "input_ids": input_ids, "input_tokens": len(input_ids),
            })
            delimiter_checks.append({"row_idx": row_index, "prompt_id": prompt_id, "actual_context_separator_ids": callback.sep})
    samples = {
        "schema_version": 2, "seed": 42, "args_namespace": vars(dataset_args),
        "records": records, "plans": plans, "validation": valid_records, "generation": generation,
        "sid_tokens": tokens, "indices": indices,
        "mixed_global128_task_counts": old_samples["mixed_global128_task_counts"],
        "selection_algorithm": {
            "mixed_0_and_1": "Exactly reuse previous audited seed42 task-proportional raw records and prompt/query/preference realization; tokenizer-independent",
            "tail": "Start mixed1, replace first record in each task by that task's longest nontruncated technical-eligible Qwen realization, preserving six-task counts and all global128 records",
            "representatives": "Stable actual Qwen full-token length then original local index, p10/p50/p90/max; all-data and technical-eligible versions separate",
            "generation": "Same three real users selected by unchanged raw history character length at p10/p50/max; all 17 original prompt templates",
            "formal_sampler_changed": False,
        },
        "data_boundary": report["auxiliary_data_status"],
    }
    write(arguments.out / "samples.json", samples)
    write(arguments.out / "selection.json", {"plans": plans, "algorithm": samples["selection_algorithm"], "mixed_global128_task_counts": samples["mixed_global128_task_counts"]})
    arrays = {name: numpy.asarray([entry[name] for entry in metadata], dtype=numpy.int32)
              for name in ["dataset_local_index", "prompt_id", "input_raw_tokens", "answer_standalone_tokens", "full_raw_tokens_with_eos", "full_tokens", "shift_effective_label_tokens"]}
    arrays.update({name: numpy.asarray([entry[name] for entry in metadata], dtype=numpy.bool_)
                   for name in ["technical_update_eligible", "full_truncated", "input_truncated", "answer_text_empty", "eos_only_supervision", "zero_supervision", "prompt_full_token_prefix_match"]})
    arrays["task_id"] = numpy.asarray([TASKS.index(entry["task"]) for entry in metadata], dtype=numpy.int8)
    numpy.savez_compressed(arguments.out / "workload.npz", **arrays)
    report["performance_samples"] = {
        "file": "samples.json", "distinct_training_records": len(records),
        "plan_sizes": {name: len(plan) for name, plan in plans.items() if isinstance(plan, list)},
        "mixed_task_counts": samples["mixed_global128_task_counts"],
        "principal_mixed_plans_identical_to_prior_audit": True,
        "validation_representatives": len(valid_records), "generation_prompts": len(generation),
    }
    report["generation_protocol"] = {
        "actual_test_users": len(test_dataset), "templates": 17,
        "beam_width": 20, "num_return_sequences": 20,
        "allowed_position_sizes": [len(callback.allowed[position]) for position in range(5)],
        "position_only_not_trie": True, "delimiter_checks": delimiter_checks,
        "gpu_generation_status": "NOT TESTED by CPU audit",
    }
    meta_inventory(arguments.model_path, tokenizer, report)
    assert not torch.cuda.is_initialized()
    report["cuda_initialized"] = False
    report["status"] = "PASS_CPU_WITH_FORMAL_DATA_BLOCKER"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    for field in ["model_path", "data", "original", "out", "config"]:
        setattr(arguments, field, getattr(arguments, field).resolve())
    arguments.out.mkdir(parents=True, exist_ok=True)
    assert not (arguments.out / "audit.json").exists(), "Refusing to overwrite prior audit evidence"
    started = time.monotonic()
    report = {
        "status": "RUNNING", "argv": sys.argv, "cwd": os.getcwd(), "pid": os.getpid(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "CPU-only original LC-Rec replay and Qwen tokenization; no model weights, training or recommendation metrics",
    }
    write(arguments.out / "audit.json", report)
    exit_code = 0
    try:
        run(arguments, report)
    except Exception:
        exit_code = 1
        report["status"] = "BLOCKED_CPU_AUDIT"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["resources"] = {"peak_rss_bytes": usage.ru_maxrss * 1024, "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime}
    report["wall_seconds"] = time.monotonic() - started
    report["exit_code"] = exit_code
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(arguments.out / "audit.json", report)
    write(arguments.out / "output_manifest.json", {
        str(path.relative_to(arguments.out)): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(arguments.out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"
    })
    print(json.dumps({"event": "finished", "status": report["status"], "exit_code": exit_code, "wall_seconds": report["wall_seconds"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
