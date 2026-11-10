#!/usr/bin/env python3
"""Bounded original LC-Rec 17-prompt beam20 generation interface probe.

This file is imported after a genuine fresh-base adapter reload by gpu_probe.
The original per-position SID callback and evaluator are called directly. An
OOM is recorded as a failed beam20 case; beam count is never reduced. Diagnostic
metrics from the untrained/barely-updated model are not research results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback


def run_inference_probe(*, model, tokenizer, data_args, sample_doc, config,
                        output_dir, rank, local_rank, event, guard, max_rank_seconds):
    import torch
    import torch.distributed as dist
    from collator import TestCollator
    from data import SeqRecDataset
    from prompt import all_prompt
    from evaluate import get_topk_results, get_metrics_results

    device = torch.device("cuda", local_rank)
    world = dist.get_world_size()
    if len(all_prompt["seqrec"]) != 17:
        raise AssertionError("Original LC-Rec SeqRec protocol must contain17 templates")
    data = SeqRecDataset(data_args, mode="test", sample_num=-1)
    if len(data) != 24772:
        raise AssertionError("Original Instruments test user count differs")
    collator = TestCollator(data_args, tokenizer)
    all_items = data.get_all_items()
    callback = data.get_prefix_allowed_tokens_fn(tokenizer)
    # The source dataset contains fixed-width SID strings. Selecting by original
    # history-string length gives a cheap deterministic short/middle/long proxy;
    # actual token counts are recorded for every tested template.
    ordered = sorted(range(len(data)), key=lambda i: (len(data.inter_data[i]["inters"]), i))
    selected = [ordered[int((len(ordered) - 1) * quantile)] for quantile in (.1, .5, .9)]
    labels = ("history_string_p10", "history_string_p50", "history_string_p90")
    filter_items = bool(config.get("filter_items", False))
    report = {"status": "RUNNING", "beam": 20, "test_prompt_ids": "all", "templates": 17,
        "max_new_tokens": 10, "test_users": len(data), "bounded_user_indices": selected,
        "selection": "Stable sort original test history string length with dataset index tie break; floor((N-1)*q), q=.1,.5,.9",
        "official_callback": "Unmodified BaseDataset.get_prefix_allowed_tokens_fn: legal tokens by SID position, not a strict item trie",
        "official_postprocess": "split('Response:')[-1], strip, remove spaces; score sort; no SID deduplication",
        "official_metrics": "Direct original get_topk_results/get_metrics_results; metrics only exercise the interface, never recommendation evidence",
        "official_aggregation": "Mean/min/max over per-prompt metrics across17 templates in author test.py/test_ddp.py",
        "filter_items": filter_items, "oom_beam_reduction": "Forbidden in this probe; original test_ddp.py decrements beam on OOM",
        "sampler": "Three explicit disjoint real user indices sharded by position across ranks; no full test scan",
        "full_test_sampler_audit": {"author_sampler": "DistributedSampler(drop_last=True)", "users": 24772, "world_size": world,
            "divisible": len(data) % world == 0, "dropped_users_for_this_dataset_world": len(data) % world,
            "duplicated_users_for_this_dataset_world": 0,
            "scope": "Arithmetic audit of the actual original two-rank dataset length; full sampler coverage separately inspected by CPU protocol audit"},
        "prompt_timings": [], "cases": [], "diagnostic_only": True,
        "training_state_restore": "Not tested; fresh adapter loading is independent of optimizer/scheduler/RNG recovery"}
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    started = time.monotonic()
    generation_limit = float(config.get("generation_max_seconds", 900))
    failures = []
    for prompt_id in range(17):
        guard()
        data.set_prompt(prompt_id)
        prompt_rows = []
        dist.barrier(); torch.cuda.synchronize()
        prompt_started = time.monotonic()
        with torch.no_grad():
            for position, user_index in enumerate(selected):
                if position % world != rank:
                    continue
                row = {"prompt_id": prompt_id, "dataset_user_index": user_index,
                       "length_selection": labels[position], "status": "RUNNING", "beam": 20}
                batch = output = None
                try:
                    guard()
                    raw = data[user_index]
                    inputs, targets = collator([raw])
                    row["input_text_sha256"] = hashlib.sha256(raw["input_ids"].encode()).hexdigest()
                    row["input_tokens"] = int(inputs["attention_mask"].sum())
                    row["input_width"] = int(inputs["input_ids"].shape[1])
                    row["target_sid"] = targets[0]
                    torch.cuda.synchronize(); t0 = time.monotonic()
                    batch = {key: tensor.to(device) for key, tensor in inputs.items()}
                    torch.cuda.synchronize(); row["transfer_seconds"] = time.monotonic() - t0
                    t0 = time.monotonic()
                    output = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        max_new_tokens=10, prefix_allowed_tokens_fn=callback, num_beams=20,
                        num_return_sequences=20, output_scores=True, return_dict_in_generate=True, early_stopping=True)
                    torch.cuda.synchronize(); row["generation_seconds"] = time.monotonic() - t0
                    sequences = output["sequences"]
                    scores = output["sequences_scores"]
                    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    postprocessed = [text.split("Response:")[-1].strip().replace(" ", "") for text in decoded]
                    generated = sequences[:, batch["input_ids"].shape[1]:].tolist()
                    position_legal = []
                    for sequence in generated:
                        legal = True
                        for token_position, token_id in enumerate(sequence):
                            if token_id == tokenizer.pad_token_id:
                                # Generation may pad finished beams after EOS.
                                continue
                            if token_id not in data.allowed_tokens.get(token_position, set()):
                                legal = False
                            if token_id == tokenizer.eos_token_id:
                                break
                        position_legal.append(legal)
                    topk = get_topk_results(decoded, scores.clone(), targets, 20,
                                            all_items=all_items if filter_items else None)
                    interface_metrics = get_metrics_results(topk, ["hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10"])
                    row.update(status="PASS", returned_sequences=int(sequences.shape[0]),
                        generated_token_ids=generated, generated_text=postprocessed,
                        generated_scores=[float(value) for value in scores],
                        all_position_legal=all(position_legal), position_legal=position_legal,
                        canonical_count=sum(text in all_items for text in postprocessed),
                        noncanonical_count=sum(text not in all_items for text in postprocessed),
                        duplicate_sid_count=len(postprocessed) - len(set(postprocessed)),
                        evaluator_topk=topk, evaluator_interface_metrics=interface_metrics,
                        metrics_interpretation="Diagnostic only; no trained recommendation claim")
                    if int(sequences.shape[0]) != 20 or not all(position_legal):
                        raise AssertionError("Original beam20 callback interface produced illegal position tokens or wrong sequence count")
                except Exception as error:
                    row.update(status="OOM" if isinstance(error, torch.cuda.OutOfMemoryError) else "FAIL",
                        error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                        lowered_beam=False)
                    failures.append({"prompt_id": prompt_id, "user_index": user_index, "status": row["status"]})
                finally:
                    row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(local_rank) / 1024 ** 3
                    row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(local_rank) / 1024 ** 3
                    del batch, output
                    torch.cuda.empty_cache()
                prompt_rows.append(row)
        torch.cuda.synchronize()
        wall = max_rank_seconds(time.monotonic() - prompt_started)
        gathered = [None] * world
        dist.all_gather_object(gathered, prompt_rows)
        all_cases = [row for group in gathered for row in group]
        all_cases.sort(key=lambda row: row["dataset_user_index"])
        report["cases"].extend(all_cases)
        success_times = [row["generation_seconds"] for row in all_cases if row["status"] == "PASS"]
        report["prompt_timings"].append({"prompt_id": prompt_id, "wall_max_rank_seconds": wall,
            "users_measured": len(all_cases), "successful_users": len(success_times),
            "individual_generation_seconds": success_times,
            "mean_user_generation_seconds": sum(success_times) / len(success_times) if success_times else None,
            "max_user_generation_seconds": max(success_times) if success_times else None,
            "central_full_prompt_two_rank_seconds": len(data) * sum(success_times) / len(success_times) / world if success_times else None,
            "conservative_full_prompt_two_rank_seconds": len(data) * max(success_times) / world if success_times else None,
            "extrapolation_limit": "Three history-length proxy users/template, batch1; excludes full-test data preparation/aggregation and fluctuations"})
        event("generation_prompt_complete", prompt_id=prompt_id, successful=len(success_times), cases=len(all_cases))
        # The shared event flushes parent report; persist this standalone result
        # incrementally too so a later generation failure cannot erase evidence.
        (Path(output_dir) / ("generation_rank" + str(rank) + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if time.monotonic() - started > generation_limit:
            report["deadline_reached"] = True
            break
    failures_all = [None] * world
    dist.all_gather_object(failures_all, failures)
    report["failures"] = [failure for group in failures_all for failure in group]
    report["elapsed_seconds"] = max_rank_seconds(time.monotonic() - started)
    report["templates_measured"] = len(report["prompt_timings"])
    report["status"] = "PASS" if len(report["prompt_timings"]) == 17 and not report["failures"] else "PARTIAL"
    if len(report["prompt_timings"]) == 17 and all(row["successful_users"] == 3 for row in report["prompt_timings"]):
        report["central_full_test_two_rank_seconds"] = sum(row["central_full_prompt_two_rank_seconds"] for row in report["prompt_timings"])
        report["conservative_full_test_two_rank_seconds"] = sum(row["conservative_full_prompt_two_rank_seconds"] for row in report["prompt_timings"])
    report["remaining_unmeasured_overhead"] = ["Complete24772-user decoding/preparation and evaluator aggregation", "Runtime variability", "Full test repeated dataset worker startup"]
    return report
