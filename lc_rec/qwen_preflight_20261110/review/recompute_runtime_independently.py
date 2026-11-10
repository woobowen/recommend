"""Non-author raw evidence/token/cost recount; never imports the main estimator.

Uses Python stdlib plus the already-installed NumPy only to independently
reconstruct the author's seeded validation template choices. No Torch, HF
tokenizer library, GPU, network, package installation, or main adapter import.
"""
import argparse
import ast
import collections
import datetime
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def original_prompts(path):
    prompts, current, template = [], None, None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "sft_prompt":
                template = ast.literal_eval(node.value)
            if isinstance(target, ast.Name) and target.id == "prompt":
                current = {}
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                if target.value.id == "all_prompt" and ast.literal_eval(target.slice) == "seqrec":
                    break
                if target.value.id == "prompt":
                    current[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.value.id == "seqrec_prompt" and call.func.attr == "append":
                    prompts.append(current.copy())
    assert len(prompts) == 17 and template
    return template, prompts


class ASCIIEncoder:
    def __init__(self, path, token_identity):
        raw = read(path)
        self.vocab = raw["model"]["vocab"]
        self.ranks = {tuple(x.split(" ") if isinstance(x, str) else x): i
                      for i, x in enumerate(raw["model"]["merges"])}
        # Original visible byte count is188; enumerate omitted bytes explicitly
        # to avoid depending on a tokenizer package's private helper.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        omitted = [x for x in range(256) if x not in visible]
        self.byte_to_char = {x: chr(x) for x in visible}
        self.byte_to_char.update({x: chr(256 + i) for i, x in enumerate(omitted)})
        self.added = {x["content"]: x["id"] for x in raw["added_tokens"]}
        self.added.update(zip(token_identity["sid_tokens"], token_identity["sid_ids"]))
        self.special = re.compile("(" + "|".join(re.escape(x) for x in sorted(self.added, key=len, reverse=True)) + ")")
        self.pattern = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

    @functools.lru_cache(maxsize=4096)
    def piece(self, value):
        parts = tuple(self.byte_to_char[x] for x in value.encode("ascii"))
        while len(parts) > 1:
            pairs = set(zip(parts[:-1], parts[1:]))
            best = min(pairs, key=lambda pair: self.ranks.get(pair, math.inf))
            if best not in self.ranks:
                break
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1]); i += 2
                else:
                    merged.append(parts[i]); i += 1
            parts = tuple(merged)
        return tuple(self.vocab[x] for x in parts)

    def encode(self, text):
        result = []
        for fragment in self.special.split(text):
            if not fragment:
                continue
            if fragment in self.added:
                result.append(self.added[fragment])
            else:
                pieces = self.pattern.findall(fragment)
                assert "".join(pieces) == fragment
                for piece in pieces:
                    result.extend(self.piece(piece))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    directory = TASK / "evidence/jobs" / args.job
    cpu_dir = TASK / "evidence/jobs" / ("qwen-20261110-cpu-" + args.key + "-r0") / "output/audit"
    ranks = [read(directory / "output/probe" / ("rank" + str(i) + ".json")) for i in range(2)]
    cpu = read(cpu_dir / "audit.json")
    guard = read(directory / "output/resources/resource-summary.json")
    request = read(directory / "request.json")
    controller = read(directory / "controller-query.json")
    cleanup = read(directory / "cleanup.json")
    assert all(x["status"] == "PASS_TECHNICAL" for x in ranks)
    assert controller["exit_code"] == 0 and controller["cleanup_verified"] and cleanup["verified"]
    assert guard["child_exit_code"] == 0 and guard["cleanup_verified"] and not guard["tracked_live_pids_after_cleanup"]
    assert abs(guard["charged_gpu_device_seconds"] - 2 * guard["elapsed_seconds"]) < 1e-9
    run_commit = ranks[0]["run_commit"]
    assert all(x["run_commit"] == run_commit for x in ranks) and request["capture_version"] == run_commit
    source_checks = []
    for entry in request["inputs"]:
        if not entry["path"].endswith(("/common.py", "/probe.py", "/resource_guard.py")):
            continue
        relative = entry["path"].split("/releases/" + run_commit + "/", 1)[1]
        raw = subprocess.check_output(["git", "show", run_commit + ":" + relative], cwd=REPO)
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        source_checks.append({"path": relative, "sha256": entry["sha256"], "git_run_commit_equal": True})

    letter = REPO / "lc_rec/letter/vendor/LETTER"
    interactions = read(letter / "data/Instruments/Instruments.inter.json")
    index = read(letter / "data/Instruments/Instruments.index.json")
    template, prompts = original_prompts(letter / "LETTER-LC-Rec/prompt.py")
    train_rows, test_rows, valid_rows = [], [], []
    rng = np.random.RandomState(42)
    for uid, raw_items in interactions.items():
        sequence = ["".join(index[str(i)]) for i in raw_items]
        for at in range(1, len(sequence) - 2):
            train_rows.append({"item": sequence[at], "inters": ", ".join(sequence[max(0, at - 20):at]), "user": uid})
        valid_row = {"item": sequence[-2], "inters": ", ".join(sequence[:-2][-20:]), "user": uid}
        for prompt in rng.choice(17, 2, replace=False):
            valid_rows.append((valid_row, int(prompt)))
        test_rows.append({"item": sequence[-1], "inters": ", ".join(sequence[:-1][-20:]), "user": uid})
    assert (len(train_rows), len(valid_rows), len(test_rows)) == (131837, 49544, 24772)

    def texts(row, prompt_id):
        instruction = prompts[prompt_id]["instruction"].format(**row)
        return template.format(instruction=instruction, response=""), template.format(instruction=instruction, response=row["item"])

    encoder = ASCIIEncoder(args.tokenizer, cpu["tokenizer"])
    eos = cpu["tokenizer"]["special_tokens_map"]["eos_token"]
    for sample in read(cpu_dir / "samples.json"):
        assert encoder.encode(sample["full_text"] + eos) == sample["input_ids"]
        prefix = encoder.encode(sample["input_text"])
        full = encoder.encode(sample["full_text"] + eos)
        assert [-100] * len(prefix) + full[len(prefix):] == sample["labels"]

    r = ranks[0]
    updates, all_rows, central_rows = [], [], []
    center_indices = [x["index"] for x in r["updates"] if x["kind"] == "mixed" and x["index"] > 0]
    if not center_indices:
        center_indices = [x["index"] for x in r["updates"] if x["nonzero_lr"]]
    for ordinal, first in enumerate(r["updates"]):
        effective, positions, supervised, samples = 0, 0, 0, 0
        for rank in ranks:
            current = rank["updates"][ordinal]
            assert current["index"] == ordinal and current["seconds"] == first["seconds"]
            for micro in current["microsteps"]:
                lengths = []
                for entry in micro["rows"]:
                    row = train_rows[entry["row_idx"]]
                    prefix, full = texts(row, entry["prompt_id"])
                    full_ids = encoder.encode(full + eos); prefix_ids = encoder.encode(prefix)
                    assert full_ids[:len(prefix_ids)] == prefix_ids and len(full_ids) - len(prefix_ids) == 5
                    lengths.append(len(full_ids)); all_rows.append(entry)
                    if ordinal in center_indices:
                        central_rows.append(entry)
                assert micro["input_shape"] == [len(lengths), max(lengths)]
                effective += sum(lengths); positions += len(lengths) * max(lengths)
                supervised += 5 * len(lengths); samples += len(lengths)
            assert sum(m["boundary"] for m in current["microsteps"]) == 1
            assert current["microsteps"][-1]["boundary"]
            assert current["adam_step_values"] == [ordinal + 1]
            assert current["engine_step_before"] == ordinal
            assert current["microsteps"][-1]["global_steps_after"] == ordinal + 1
            assert current["nonzero_lr"] == any(v > 0 for v in current["lr_before"])
        assert samples == 64 and supervised == 320
        for rank in ranks:
            current = rank["updates"][ordinal]
            assert (effective, positions, supervised, samples) == tuple(current[k] for k in ["global_effective_tokens", "global_padded_tokens", "global_supervised_tokens", "global_samples"])
        updates.append({"index": ordinal, "kind": first["kind"], "global_samples": samples,
                        "effective_tokens_reencoded": effective, "tensor_positions_recounted": positions,
                        "padding_positions_recounted": positions - effective, "supervised_targets_recounted": supervised,
                        "seconds_MAX_rank": first["seconds"], "nonzero_lr": first["nonzero_lr"]})

    parameter_bytes = 0
    width = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "int8": 1, "uint8": 1, "int64": 8}
    trainable_names = {name for name, row in r["parameter_inventory"].items() if row["requires_grad"]}
    trainable_elements = sum(math.prod(r["parameter_inventory"][n]["shape"]) for n in trainable_names)
    assert trainable_elements == r["trainable"]["parameters"]
    if r["route"] == "F":
        assert trainable_names == set(r["parameter_inventory"])
        assert trainable_elements == cpu["model"]["expanded_unique_parameters"]
        assert r["full_parameter_scope"]["all_instantiated_parameters_trainable"]
        assert r["full_parameter_scope"]["official_text_class_only"] == (args.key == "q35")
        assert not any("visual" in n or n.startswith("mtp.") for n in trainable_names)
    for row in r["parameter_inventory"].values():
        assert math.prod(row["shape"]) == row["numel"]
        parameter_bytes += row["numel"] * width[row["dtype_after_ds"].removeprefix("torch.")]
    expected_calls = len(updates) * r["training"]["accumulation"]
    gradient_checks = []
    for rank in ranks:
        assert set(rank["gradient_observations"]) == trainable_names
        assert set(n for g in rank["optimizer"]["groups"] for n in g["names"]) == trainable_names
        assert all(o["calls"] == expected_calls and o["finite"] for o in rank["gradient_observations"].values())
        assert all(o["nonzero_calls"] > 0 for n, o in rank["gradient_observations"].items() if "lora_" in n)
        assert all(x["changed_elements"] > 0 for name, x in rank["parameter_changes"].items() if "embed_tokens" in name)
        if r["route"] == "L":
            assert all(x["changed_elements"] > 0 for x in rank["parameter_changes"].values())
        else:
            assert any(x["changed_elements"] > 0 for name, x in rank["parameter_changes"].items() if "embed_tokens" not in name)
        assert rank["activation_gradients"]["actual_input_sid_position_gradient_abs_max"] > 0
        assert rank["activation_gradients"]["actual_supervised_sid_logit_gradient_abs_max"] > 0
        assert rank["temperature_ce_independent_abs_diff"] < 2e-5
        for state in rank["optimizer"]["state_layout"]:
            assert state["param_dtype"] == "torch.float32" and state["param_device"].startswith("cuda:")
            assert all(state["state"][k]["dtype"] == "torch.float32" and state["state"][k]["device"].startswith("cuda:") for k in ["exp_avg", "exp_avg_sq"])
        assert rank["all_rank_parameter_hashes"][0] == rank["all_rank_parameter_hashes"][1]
        gradient_checks.append({"rank": rank["rank"], "trainable_tensors": len(trainable_names),
                                "calls_per_tensor": expected_calls, "actual_nonzero_lr_updates": sum(x["nonzero_lr"] for x in rank["updates"]),
                                "finite_and_gradient_counts_pass": True, "SID_paths_nonzero": True,
                                "Adam_master_and_moments_gpu_fp32": True})

    validation = []
    for first in r["validation"]:
        lengths = [len(encoder.encode(texts(*valid_rows[i])[1] + eos)) for i in first["indices"]]
        assert first["global_rows"] == len(lengths) == 16
        assert first["global_tokens"] == sum(lengths)
        assert first["global_padding"] == 8 * max(lengths[:8]) + 8 * max(lengths[8:]) - sum(lengths)
        validation.append({"label": first["label"], "tokens_reencoded": sum(lengths),
                           "padding_recounted": first["global_padding"], "seconds": first["seconds_including_collator"]})

    token_to_id = dict(zip(cpu["tokenizer"]["sid_tokens"], cpu["tokenizer"]["sid_ids"]))
    id_to_token = {v: k for k, v in token_to_id.items()}
    position_sets = [{token_to_id[s[pos]] for s in index.values()} for pos in range(4)]
    canonical = {"".join(parts) for parts in index.values()}
    strata, user_ids, noncanonical, beams_checked = [], set(), 0, 0
    for label in ["short", "middle", "long", "boundary"]:
        pair = [x for rank in ranks for x in rank["beam20"] if x["label"] == label]
        assert len(pair) == 2 and pair[0]["seconds_two_rank_max"] == pair[1]["seconds_two_rank_max"]
        for beam_row in pair:
            row = test_rows[beam_row["row_idx"]]
            prompt, _ = texts(row, 0)
            assert row["user"] == beam_row["user_id"] and sha_bytes(prompt.encode()) == beam_row["input_text_sha256"]
            assert len(encoder.encode(prompt)) == beam_row["input_tokens"]
            user_ids.add(row["user"])
            assert len(beam_row["sequences"]) == len(beam_row["sequence_scores"]) == 20
            assert set(beam_row["callback_calls"]) == set(map(str, range(5)))
            assert all(v > 0 for v in beam_row["callback_calls"].values())
            assert all(math.isfinite(v) for v in beam_row["sequence_scores"])
            valid_canonical = 0
            for sequence in beam_row["sequences"]:
                ids = sequence["token_ids"]
                assert ids[4] == cpu["tokenizer"]["eos_token_id"]
                assert all(ids[pos] in position_sets[pos] for pos in range(4))
                decoded = "".join(id_to_token[v] for v in ids[:4])
                assert decoded == sequence["sid"] and sequence["canonical"] == (decoded in canonical)
                valid_canonical += decoded in canonical; beams_checked += 1
            assert valid_canonical / 20 == beam_row["canonical_fraction"]
            noncanonical += 20 - valid_canonical
        strata.append({"label": label, "mean_input_tokens": sum(x["input_tokens"] for x in pair) / 2,
                       "dual_user_seconds": pair[0]["seconds_two_rank_max"],
                       "seconds_per_user": pair[0]["seconds_two_rank_max"] / 2})
    assert len(user_ids) == 8

    steps_per_epoch = -(-r["formal_loader"]["length_per_rank"] // r["training"]["accumulation"])
    formal_steps = 4 * steps_per_epoch
    assert formal_steps == r["training"]["formal_updates"]
    chosen = [x for x in updates if x["index"] in center_indices]
    chosen_seconds = sum(x["seconds_MAX_rank"] for x in chosen)
    chosen_tokens = sum(x["effective_tokens_reencoded"] for x in chosen)
    chosen_positions = sum(x["tensor_positions_recounted"] for x in chosen)
    observed_mean = chosen_tokens / (64 * len(chosen))
    hist = {int(k): v for k, v in cpu["lengths"]["train"]["histogram"].items()}
    target_mean = sum(k * v for k, v in hist.items()) / sum(hist.values())
    scaling = max(1., target_mean / observed_mean)
    per_update_center = chosen_seconds / len(chosen) * scaling
    per_update_conservative = max(x["seconds_MAX_rank"] for x in updates[1:]) * scaling
    cold_excess = max(0., updates[0]["seconds_MAX_rank"] - chosen_seconds / len(chosen))
    assigned = {x["label"]: 0 for x in strata}
    test_seconds = 0.
    for token_length, count in sorted((int(k), v) for k, v in cpu["lengths"]["test_prompt0"]["histogram"].items()):
        selected = sorted(strata, key=lambda row: (abs(row["mean_input_tokens"] - token_length), row["mean_input_tokens"]))[0]
        assigned[selected["label"]] += count
        test_seconds += count * selected["seconds_per_user"]
    assert sum(assigned.values()) == 24772
    if r["route"] == "L":
        native_bytes = r["adapter_save"]["file_bytes"]; native_seconds = r["adapter_save"]["seconds"]
        restore = max(x["adapter_reload_seconds"] for x in ranks)
        extra_init = max(x["inference_base_reload_seconds"] for x in ranks)
        assert all(x["adapter_reload"]["all_keys_and_values_exact"] and x["adapter_reload"]["sharing"]["same_parameter_object"] for x in ranks)
        native_evidence = {"scope": "LoRA adapter plus complete shared active embedding/head",
                           "serialized_file_sha256": r["adapter_save"]["sha256"],
                           "serialized_tensor_keys": len(r["adapter_save"]["keys"]),
                           "strict_reload_values_all_ranks": True}
    else:
        native_bytes = sum(x["bytes"] for x in r["full_model_save"]["files"]); native_seconds = r["full_model_save"]["seconds"]
        restore = max(x["full_model_reload_seconds"] for x in ranks); extra_init = 0.
        assert all(x["full_model_reload"]["all_keys_shapes_dtypes_values_exact"] for x in ranks)
        states = r["full_model_save"]["parameter_state"]
        embedding_key = next(n for n in states if n.endswith("embed_tokens.weight"))
        assert states[embedding_key]["sha256"] == states["lm_head.weight"]["sha256"]
        assert all(x["full_model_reload"]["state_tensors"] == len(states) for x in ranks)
        assert all(x["full_model_reload"]["sharing"]["same_parameter_object"] for x in ranks)
        native_evidence = {"scope": "all instantiated official text-model parameters, model-only save",
                           "full_parameter_scope": r["full_parameter_scope"],
                           "serialized_weight_files": r["full_model_save"]["files"],
                           "serialized_state_keys": len(states), "tied_embedding_head_saved_hashes_equal": True,
                           "strict_keys_shapes_dtypes_values_reload_all_ranks": True}
    assert all(not x["full_optimizer_checkpoint_written"] for x in ranks)
    native_evidence["optimizer_scheduler_rng_checkpoint_and_restore"] = "NOT TESTED / NOT WRITTEN"
    checkpoint_bytes = 2 * parameter_bytes + 12 * trainable_elements + 2 ** 30
    bandwidth = native_bytes / native_seconds
    initialization = next(x["elapsed_seconds"] for x in r["stages"] if x["stage"] == "bounded_training") + extra_init + cold_excess
    launch_overhead = max(0., guard["elapsed_seconds"] - max(x["elapsed_seconds"] for x in ranks))
    center = {"train": formal_steps * per_update_center,
              "validation": 4 * len(valid_rows) * next(x["seconds"] for x in validation if x["label"] == "mixed") / 16,
              "checkpoint": 4 * checkpoint_bytes / bandwidth + native_seconds,
              "best_restore": restore, "beam20_test": test_seconds, "init": initialization}
    conservative = {"train": formal_steps * per_update_conservative,
                    "validation": 4 * len(valid_rows) * max(x["seconds"] for x in validation) / 16,
                    "checkpoint": 4 * checkpoint_bytes * 4 / bandwidth + 4 * 60 + 2 * native_seconds,
                    "best_restore": 2 * restore, "beam20_test": len(test_rows) * max(x["seconds_per_user"] for x in strata),
                    "init": 2 * initialization}
    center["overhead"] = sum(center.values()) / 10 + launch_overhead
    conservative["overhead"] = sum(conservative.values()) / 4 + launch_overhead
    status = "UNLIKELY_WITHIN_96H" if sum(conservative.values()) > 96 * 3600 else "LIKELY_WITHIN_96H" if r["route"] == "L" and len(center_indices) >= 2 and sum(conservative.values()) < .8 * 96 * 3600 else "INCONCLUSIVE"
    result = {"schema": "independent_qwen_raw_runtime_cost_v1", "reviewer": "/root/letter_semantics",
              "job": args.job, "key": args.key, "route": r["route"], "run_commit": run_commit,
              "status": "PASS_RAW_RUNTIME_AND_INDEPENDENT_CONDITIONAL_ARITHMETIC", "formal_seed_confirmed": False,
              "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "review_environment": {"interpreter": sys.executable, "python": sys.version, "numpy": np.__version__, "GPU_execution": False},
              "script_sha256": sha(Path(__file__)), "source_checks": source_checks,
              "raw_input_sha256": {str(q.relative_to(TASK)): sha(q) for q in [directory / "request.json", directory / "output/probe/rank0.json", directory / "output/probe/rank1.json", directory / "output/resources/resource-summary.json", cpu_dir / "audit.json", args.tokenizer] if q.is_relative_to(TASK)},
              "independent_encoder": {"official_tokenizer_path": str(args.tokenizer), "sha256": sha(args.tokenizer), "method": "Own ASCII regex, GPT2 byte map and ranked BPE; no HF tokenizer/main adapter used", "cpu_samples_exact": 20},
              "updates": updates, "gradient_checks": gradient_checks, "validation": validation,
              "sampling": {"presentations": len(all_rows), "unique_train_rows": len({x["row_idx"] for x in all_rows}), "unique_row_prompt_pairs": len({(x["row_idx"], x["prompt_id"]) for x in all_rows}), "central_presentations": len(central_rows), "central_unique_rows": len({x["row_idx"] for x in central_rows})},
              "throughput": {"central_indices": center_indices, "central_seconds": chosen_seconds, "central_effective_tokens": chosen_tokens, "central_tensor_positions": chosen_positions, "central_padding_positions": chosen_positions - chosen_tokens, "samples_per_second": 64 * len(chosen) / chosen_seconds, "effective_tokens_per_second": chosen_tokens / chosen_seconds},
              "formal_work": {"loader_batches_per_rank": r["formal_loader"]["length_per_rank"], "updates_per_epoch": steps_per_epoch, "four_epoch_updates": formal_steps, "original_rows_four_epochs": 527348, "global64_equivalent_presentations": 64 * formal_steps, "validation_presentations": 4 * len(valid_rows), "test_users": len(test_rows)},
              "native_save_reload": {"file_bytes": native_bytes, "save_seconds": native_seconds, "write_bytes_per_second": bandwidth, "best_restore_proxy_seconds_MAX_rank": restore, "extra_inference_base_load_seconds_MAX_rank": extra_init},
              "native_save_reload_identity": native_evidence,
              "beam_review": {"users": len(user_ids), "beams": beams_checked, "all_positional_tokens_and_eos_pass": True, "noncanonical_beams": noncanonical, "strata": strata, "histogram_users_by_stratum": assigned},
              "cost_inputs": {"measured_mean_tokens": observed_mean, "full_mean_tokens": target_mean, "upward_token_scale": scaling, "cold_excess_once": cold_excess, "current_parameter_bytes": parameter_bytes, "trainable_elements": trainable_elements, "checkpoint_proxy_bytes": checkpoint_bytes, "launcher_teardown_seconds": launch_overhead},
              "center": {"seconds_by_component": center, "total_seconds": sum(center.values()), "hours": sum(center.values()) / 3600, "days": sum(center.values()) / 86400},
              "conservative": {"seconds_by_component": conservative, "total_seconds": sum(conservative.values()), "hours": sum(conservative.values()) / 3600, "days": sum(conservative.values()) / 86400},
              "96h_status": status, "resource_receipt": guard,
              "limitations": ["No second GPU execution or independent recomputation from unretained gradient/logit arrays; counters and equality checks are verified against raw logs and reviewed code.", "Checkpoint byte/time formulas are explicit assumptions; optimizer/scheduler/RNG checkpoint and formal restore were not written or tested.", "Generation CPU decoding/aggregation and sustained four-epoch behavior remain unknown; overhead percentages are assumptions.", "No recommendation metrics or formal model-quality ranking."]}
    reconciliation = TASK / "config/update_reconciliations.json"
    if reconciliation.exists():
        certified = []
        for record in read(reconciliation)["certified_failed_zero_update_jobs"]:
            if record["key"] != args.key or record["route"] != r["route"]:
                continue
            original = TASK / "evidence/jobs" / record["job_id"]
            for relative, expected_sha in record["raw_files_sha256"].items():
                assert sha(original / relative) == expected_sha
            review = TASK / record["independent_review_path"]
            assert sha(review) == record["independent_review_sha256"]
            failed = [read(original / "output/probe" / ("rank" + str(i) + ".json")) for i in range(2)]
            for failed_rank in failed:
                assert failed_rank["updates"] == []
                pending = failed_rank["current_update"]
                assert pending["index"] == pending["engine_step_before"] == 0 and not any(pending["lr_before"])
                assert all(m["global_steps_after"] == 0 and not m["boundary"] for m in pending["microsteps"])
            prior_resources = read(original / "output/resources/resource-summary.json")
            assert prior_resources["child_exit_code"] != 0 and prior_resources["cleanup_verified"]
            certified.append({"job_id": record["job_id"], "completed_global_updates": 0,
                              "failed_attempt_retained": True, "precharged_attempts_retained": record["attempts_retained"],
                              "prior_device_seconds_still_charged": prior_resources["charged_gpu_device_seconds"],
                              "certification_config_sha256": sha(reconciliation),
                              "non_author_source_failure_review_sha256": sha(review),
                              "raw_file_hashes_verified": True,
                              "partial_Adam_state_not_reused": "Retry starts a new job from original model, seed and warmup; first successful update is again LR0"})
        if certified:
            result["prior_failed_attempt_reconciliation"] = certified
    if args.summary:
        main_output = read(args.summary)
        published = main_output["models"][args.key]["routes"][r["route"]]["seed_estimate"]
        differences = {}
        for mode, own in [("center", center), ("conservative", conservative)]:
            for name, value in own.items():
                delta = abs(value - published[mode]["seconds_by_component"][name]); differences[mode + "." + name] = delta
                assert delta < 1e-6, (mode, name, value, published[mode]["seconds_by_component"][name])
        assert status == published["status"]
        assert parameter_bytes == published["checkpoint_proxy"]["current_parameter_bytes_unique_parameter_inventory"]
        assert checkpoint_bytes == published["checkpoint_proxy"]["bytes_per_checkpoint_assumption"]
        assert formal_steps == published["workload"]["effective_formal_optimizer_steps"]
        assert assigned == published["measured"]["test_histogram_assigned_users"]
        assert result["sampling"]["unique_train_rows"] == published["measured"]["sampling"]["actual_unique_train_rows"]
        for mode in ["center", "conservative"]:
            assert abs(result[mode]["total_seconds"] - published[mode]["total_seconds"]) < 1e-6
        result["main_summary_comparison"] = {"sha256": sha(args.summary),
            "main_analysis_identity": main_output.get("analysis"),
            "scope": args.key + " " + r["route"] + " only; independently derived values precede comparison",
            "all_components_match_within_1e6_seconds": True, "absolute_differences": differences,
            "parameter_checkpoint_bytes_steps_histogram_sampling_and_totals_match": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"job": args.job, "center_hours": result["center"]["hours"], "conservative_hours": result["conservative"]["hours"], "96h_status": status, "report_sha256": sha(args.output)}))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
