"""CPU-only audit of the installed Trainer schedule and two-rank sampler tail."""

from __future__ import annotations

import argparse
from array import array
import datetime as datetime_module
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


TRAIN_RECORDS = 221225
VALID_RECORDS = 49544
WORLD_SIZE = 2
GLOBAL_BATCH = 128
EPOCHS = 4


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def identity(value):
    result = {"class": type(value).__module__ + "." + type(value).__name__}
    for name in ("batch_size", "drop_last", "even_batches", "split_batches", "num_processes", "process_index"):
        attribute = getattr(value, name, None)
        if isinstance(attribute, (str, int, float, bool, type(None))):
            result[name] = attribute
    return result


def source_record(function, needles, output):
    source_lines, start_line = inspect.getsourcelines(function)
    source = "".join(source_lines)
    output.write_text(source)
    selected = []
    for offset, line in enumerate(source_lines):
        if any(needle in line for needle in needles):
            selected.append({"line": start_line + offset, "text": line.rstrip()})
    return {
        "symbol": function.__qualname__,
        "installed_file": inspect.getsourcefile(function),
        "first_line": start_line,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_file": output.name,
        "control_lines": selected,
    }


def audit(output, run_commit, report):
    import torch
    from accelerate import Accelerator
    from accelerate.data_loader import prepare_data_loader
    from torch.utils.data import DataLoader
    from transformers import DefaultFlowCallback, Trainer, TrainerControl, TrainerState, TrainingArguments
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA_VISIBLE_DEVICES='' is required"
    assert not torch.cuda.is_initialized()
    assert importlib.metadata.version("transformers") == "4.57.6"
    assert importlib.metadata.version("accelerate") == "1.12.0"
    Accelerator(cpu=True)
    report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")}
    report["cuda_initialized_before"] = torch.cuda.is_initialized()
    report["sources"] = {}
    source_requests = [
        ("initial_values", Trainer.set_initial_training_values, ("num_update_steps_per_epoch", "gradient_accumulation_steps", "max_steps =", "math.ceil")),
        ("inner_training_loop", Trainer._inner_training_loop, ("steps_in_epoch", "remainder", "total_updates", "num_batches", "current_gradient_accumulation_steps", "do_sync_step", "state.epoch", "on_epoch_end", "_load_best_model")),
        ("epoch_callback", DefaultFlowCallback.on_epoch_end, ("eval_strategy", "eval_delay", "should_evaluate", "save_strategy", "should_save")),
        ("step_callback", DefaultFlowCallback.on_step_end, ("global_step", "max_steps", "should_training_stop", "should_save")),
        ("save_checkpoint", Trainer._save_checkpoint, ("save_model", "save_only_model", "_save_optimizer_and_scheduler", "_save_scaler", "_save_rng_state", "save_to_json", "_rotate_checkpoints")),
        ("best_restore", Trainer._load_best_model, ("deepspeed", "checkpoint", "load_adapter", "load_state_dict", "strict=")),
        ("prepare_data_loader", prepare_data_loader, ("even_batches", "use_seedable_sampler", "data_seed", "BatchSamplerShard")),
        ("get_batch_samples", Trainer.get_batch_samples, ("num_batches", "next(epoch_iterator)", "_get_num_items_in_batch")),
        ("count_items", Trainer._get_num_items_in_batch, ("model_accepts_loss_kwargs", "compute_loss_func", "num_items_in_batch", "average_tokens_across_devices")),
        ("compute_loss", Trainer.compute_loss, ("model_accepts_loss_kwargs", "num_items_in_batch", "outputs = model", "average_tokens_across_devices")),
        ("training_step", Trainer.training_step, ("model_accepts_loss_kwargs", "current_gradient_accumulation_steps", "scale_wrt_gas", "backward")),
        ("qwen_forward", Qwen3ForCausalLM.forward, ("**kwargs", "labels", "loss_function")),
    ]
    for name, function, needles in source_requests:
        report["sources"][name] = source_record(function, needles, output / (name + ".py.txt"))
    loop_source = inspect.getsource(Trainer._inner_training_loop)
    for marker in (
        "remainder = steps_in_epoch % args.gradient_accumulation_steps",
        "current_gradient_accumulation_steps = len(batch_samples)",
        "(step + 1) == steps_in_epoch",
        "state.epoch = epoch + (step + 1) / steps_in_epoch",
    ):
        assert marker in loop_source, "Installed Trainer control flow changed: " + marker

    report["layouts"] = []
    for micro in (8, 4, 2, 1):
        accumulation = GLOBAL_BATCH // (WORLD_SIZE * micro)
        args = TrainingArguments(
            output_dir=str(output / "unused_trainer_output"),
            use_cpu=True,
            seed=42,
            data_seed=42,
            per_device_train_batch_size=micro,
            per_device_eval_batch_size=micro,
            gradient_accumulation_steps=accumulation,
            num_train_epochs=EPOCHS,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.01,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            eval_delay=1,
            save_total_limit=5,
            load_best_model_at_end=True,
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        trainer = object.__new__(Trainer)
        trainer.args = args
        trainer.train_dataset = range(TRAIN_RECORDS)
        trainer.processing_class = None
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None
        ranks = []
        presentation_counts = array("I", [0]) * TRAIN_RECORDS
        initial_values = None
        for rank in range(WORLD_SIZE):
            torch.manual_seed(42)
            sampler = trainer._get_train_sampler()
            raw_loader = DataLoader(
                trainer.train_dataset,
                batch_size=micro,
                sampler=sampler,
                drop_last=False,
                collate_fn=lambda entries: {"record_indices": torch.tensor(entries, dtype=torch.long)},
            )
            prepared = prepare_data_loader(
                raw_loader,
                device=torch.device("cpu"),
                num_processes=WORLD_SIZE,
                process_index=rank,
                split_batches=False,
                put_on_device=False,
                rng_types=[],
                even_batches=True,
                use_seedable_sampler=True,
                data_seed=42,
            )
            prepared.set_epoch(0)
            values = trainer.set_initial_training_values(args, prepared, GLOBAL_BATCH)
            if initial_values is None:
                initial_values = values
            else:
                assert values == initial_values
            steps_in_epoch = len(prepared)
            remainder = steps_in_epoch % accumulation
            if remainder == 0:
                remainder = accumulation
            total_updates = steps_in_epoch // accumulation + int(remainder < accumulation)
            iterator = iter(prepared)
            row_ids = array("I")
            last_sizes = []
            observed_updates = 0
            batches_seen = 0
            observed_tail_samples = None
            for update_index in range(total_updates):
                requested = accumulation if update_index != total_updates - 1 else remainder
                batches, num_items = trainer.get_batch_samples(iterator, requested, torch.device("cpu"))
                assert len(batches) == requested and num_items is None
                samples_this_update = 0
                for batch in batches:
                    entries = batch["record_indices"].tolist()
                    batches_seen += 1
                    samples_this_update += len(entries)
                    row_ids.extend(entries)
                    last_sizes.append(len(entries))
                    last_sizes = last_sizes[-accumulation:]
                    for record_index in entries:
                        presentation_counts[record_index] += 1
                do_sync_step = batches_seen % accumulation == 0 or batches_seen == steps_in_epoch
                assert do_sync_step
                observed_updates += 1
                if update_index == total_updates - 1:
                    observed_tail_samples = samples_this_update
            assert next(iterator, None) is None
            assert batches_seen == len(prepared)
            assert observed_updates == total_updates
            ranks.append({
                "rank": rank,
                "raw_loader": identity(raw_loader),
                "prepared_loader": identity(prepared),
                "prepared_batch_sampler": identity(prepared.batch_sampler),
                "sampler": identity(prepared.sampler),
                "actual_iterated_batches": batches_seen,
                "presentations": len(row_ids),
                "unique_records": len(set(row_ids)),
                "first_eight_indices": list(row_ids[:8]),
                "last_eight_indices": list(row_ids[-8:]),
                "ordered_indices_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(),
                "index_hash_encoding": "native-endian unsigned32 array; host byteorder=" + sys.byteorder,
                "last_microbatch_sizes": last_sizes,
                "actual_get_batch_samples_calls": observed_updates,
                "last_update_microbatches": remainder,
                "last_update_local_presentations": observed_tail_samples,
            })
        num_epochs, updates_per_epoch, num_examples, nominal_samples, epoch_based, loader_length, max_steps = initial_values
        assert num_epochs == EPOCHS and epoch_based
        assert num_examples == TRAIN_RECORDS
        assert sum(count == 0 for count in presentation_counts) == 0
        state = TrainerState(max_steps=max_steps, logging_steps=args.logging_steps)
        flow = DefaultFlowCallback()
        epoch_trace = []
        for epoch in range(num_epochs):
            before_step = state.global_step
            consumed_microbatches = 0
            executed_last_window = 0
            for update_index in range(ranks[0]["actual_get_batch_samples_calls"]):
                executed_last_window = accumulation if update_index != total_updates - 1 else remainder
                consumed_microbatches += executed_last_window
                state.global_step += 1
                state.epoch = epoch + consumed_microbatches / loader_length
                control = flow.on_step_end(args, state, TrainerControl())
                if control.should_epoch_stop or control.should_training_stop:
                    break
            control = flow.on_epoch_end(args, state, control)
            epoch_trace.append({
                "epoch_index": epoch,
                "state_epoch": state.epoch,
                "global_step": state.global_step,
                "updates": state.global_step - before_step,
                "microbatches_per_rank": consumed_microbatches,
                "complete_epoch_tail_reached": consumed_microbatches == loader_length,
                "last_executed_update_microbatches_per_rank": executed_last_window,
                "last_executed_update_global_presentations": executed_last_window * micro * WORLD_SIZE,
                "tail_microbatches_per_rank": ranks[0]["last_update_microbatches"] if consumed_microbatches == loader_length else 0,
                "tail_global_presentations": sum(rank["last_update_local_presentations"] for rank in ranks) if consumed_microbatches == loader_length else 0,
                "global_presentations": consumed_microbatches * micro * WORLD_SIZE,
                "unstepped_microbatches_after_epoch": 0,
                "should_evaluate": control.should_evaluate,
                "should_save": control.should_save,
                "should_training_stop": control.should_training_stop,
            })
            if control.should_training_stop:
                break
        assert state.global_step == max_steps
        presentations = sum(rank["presentations"] for rank in ranks)
        report["layouts"].append({
            "micro": micro,
            "accumulation": accumulation,
            "world_size": WORLD_SIZE,
            "full_global_batch": GLOBAL_BATCH,
            "ranks": ranks,
            "trainer_set_initial_training_values": {
                "num_train_epochs": num_epochs,
                "num_update_steps_per_epoch": updates_per_epoch,
                "num_examples": num_examples,
                "num_train_samples_nominal": nominal_samples,
                "epoch_based": epoch_based,
                "len_dataloader": loader_length,
                "max_steps": max_steps,
            },
            "warmup_steps_actual_method": args.get_warmup_steps(max_steps),
            "epoch_loop_control": epoch_trace,
            "base_records_four_epochs": TRAIN_RECORDS * EPOCHS,
            "actual_sampler_presentations_per_epoch": presentations,
            "sampler_repeated_presentations_per_epoch": presentations - TRAIN_RECORDS,
            "actual_sampler_presentations_four_epochs": sum(event["global_presentations"] for event in epoch_trace),
            "full_sampler_presentations_if_four_complete_epochs": presentations * EPOCHS,
            "scheduled_global128_equivalent_presentations": max_steps * GLOBAL_BATCH,
            "difference_due_to_partial_epoch_tail": max_steps * GLOBAL_BATCH - sum(event["global_presentations"] for event in epoch_trace),
            "effective_validation_calls": sum(event["should_evaluate"] for event in epoch_trace),
            "validation_records_per_call": VALID_RECORDS,
            "validation_total_records": VALID_RECORDS * sum(event["should_evaluate"] for event in epoch_trace),
            "scheduled_full_checkpoint_calls": sum(event["should_save"] for event in epoch_trace),
            "final_explicit_model_save_calls": 1,
            "best_model_restore_calls": 1,
            "training_arguments": {
                "eval_strategy": args.eval_strategy.value,
                "save_strategy": args.save_strategy.value,
                "eval_delay": args.eval_delay,
                "load_best_model_at_end": args.load_best_model_at_end,
                "save_total_limit": args.save_total_limit,
                "save_only_model": args.save_only_model,
                "metric_for_best_model": args.metric_for_best_model,
                "greater_is_better": args.greater_is_better,
                "average_tokens_across_devices": args.average_tokens_across_devices,
            },
        })
        write_json(output / "protocol.json", report)
    report["loss_normalization_boundary"] = {
        "original_lc_rec": "Each rank microbatch causal shifted CE is a mean over its valid labels; accumulation/rank averaging does not equal a full-window token mean when answer lengths differ.",
        "current_trainer_default_risk": "Qwen3.forward accepts **kwargs; Trainer can pass num_items_in_batch across the accumulation window. Native 4.57.6 Trainer loss scaling must not be called equivalent to original 4.31 LC-Rec without an explicit adapter.",
        "this_cpu_probe": "The uninitialized Trainer object has model_accepts_loss_kwargs=False and compute_loss_func=None only for index iteration; no loss, model, optimizer, backward or training executed.",
        "required_bounded_gpu_scope": "Explicit per-micro CE with no temperature and no num_items_in_batch; original rank/micro averaging must be evidenced independently.",
        "epoch_tail_limit": "Native Trainer sets current_gradient_accumulation_steps to actual final chunk size. The bounded DeepSpeed loop uses full global128; exact short-tail numerical behavior with DeepSpeed is NOT TESTED by this CPU protocol audit.",
    }
    report["checkpoint_scope"] = {
        "checkpoint_policy": "Four epoch checkpoints plus final save_model; save_total_limit=5; lowest eval_loss; load_best_model_at_end.",
        "full_state_by_source": "save_only_model=False invokes model, optimizer/scheduler, scaler as applicable, RNG and Trainer JSON state saving. DeepSpeed-specific state layout is not written here.",
        "not_measured": ["Model loading", "Training or loss", "Full optimizer checkpoint I/O", "Optimizer/scheduler/RNG recovery", "DeepSpeed epoch-tail gradients", "Real model selection and best restoration"],
    }
    report["limitations"] = [
        "CPU sharding traverses only dummy integer record IDs in one epoch per rank; all four epoch batch-size and callback traces follow inspected installed control flow, not a model run.",
        "prepare_data_loader is explicitly parameterized for two ranks on one CPU process; no distributed CPU process group or GPU is created.",
        "Sampler identities and tail sizes are real Accelerate 1.12 outputs, but Dataset prompt/query/preference RNG behavior and communication are outside this probe.",
        "The dummy Trainer object bypasses __init__ to avoid model/optimizer initialization; actual set_initial_training_values, sampler, get_batch_samples and DefaultFlowCallback methods are invoked.",
    ]
    report["cuda_initialized_after"] = torch.cuda.is_initialized()
    assert not report["cuda_initialized_after"]
    report["status"] = "PASS_CPU_PROTOCOL_ONLY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    args = parser.parse_args()
    if len(args.run_commit) != 40 or any(character not in "0123456789abcdef" for character in args.run_commit):
        parser.error("--run-commit must be a full lowercase SHA")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "status": "RUNNING",
        "kind": "CPU_ONLY_INSTALLED_TRAINER_AND_ACCELERATE_PROTOCOL",
        "run_commit": args.run_commit,
        "argv": sys.argv,
        "interpreter": sys.executable,
        "cwd": str(Path.cwd()),
        "observed_utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "training_records": TRAIN_RECORDS,
    }
    write_json(args.output / "protocol.json", report)
    try:
        audit(args.output, args.run_commit, report)
    except Exception:
        report["status"] = "BLOCKED_CPU_PROTOCOL"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "protocol.json", report)
    print(json.dumps({"status": report["status"], "layouts": [{"micro": layout["micro"], "scheduled_updates": layout["trainer_set_initial_training_values"]["max_steps"], "validation_calls": layout["effective_validation_calls"]} for layout in report["layouts"]]}))


if __name__ == "__main__":
    main()
