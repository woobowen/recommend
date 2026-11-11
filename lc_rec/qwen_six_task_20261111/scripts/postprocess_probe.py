"""CPU-only replay of bounded LC-Rec beam20 outputs with a cached SID callback."""

from __future__ import annotations

import argparse
import datetime as datetime_module
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

from common import Allowed, sha, write


def record_file(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def value_digest(value):
    return text_digest(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def import_original(directory):
    directory = Path(directory).resolve()
    sys.path.insert(0, str(directory))
    modules = {}
    for name in ("prompt", "data", "collator", "evaluate"):
        specification = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        modules[name] = module
    return modules


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--probe", type=Path, action="append", default=[])
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--model-tokenizer", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-commit", default=os.environ.get("RUN_COMMIT", ""))
    parser.add_argument("--cpu-threads", type=int, default=4)
    arguments = parser.parse_args()
    if not arguments.probe and arguments.evidence is None:
        parser.error("Provide --probe DIR or --evidence DIR containing the fixed GPU rank reports")
    if not re.fullmatch("[a-f0-9]{40}", arguments.run_commit):
        parser.error("--run-commit must be the complete immutable CPU-source commit")
    if not 1 <= arguments.cpu_threads <= 8:
        parser.error("CPU replay uses between1 and8 CPU threads")
    return arguments


def probe_directories(arguments):
    found = []
    for requested in arguments.probe:
        directory = requested
        if not (directory / "rank0.json").is_file() and directory.name != "probe":
            directory = requested / "output" / "probe"
        found.append(directory.resolve())
    if arguments.evidence is not None:
        directory = arguments.evidence.resolve()
        if (directory / "rank0.json").is_file():
            found.append(directory)
        else:
            found.extend(path.parent for path in directory.rglob("rank0.json") if path.parent.name == "probe")
    unique = sorted(set(found), key=str)
    if not unique:
        raise FileNotFoundError("No bounded GPU rank reports were found")
    return unique


def route_identity(directory):
    report_path = directory / "rank0.json"
    if report_path.is_file():
        try:
            return json.loads(report_path.read_text())["route_key"]
        except (ValueError, KeyError):
            pass
    for part in reversed(directory.parts):
        match = re.fullmatch("lc-qwen6-20261111-(q06|q17)-([lf])-r[0-9]+", part)
        if match:
            return match.group(1) + "-" + match.group(2).upper()
    return "unidentified-" + text_digest(str(directory))[:12]


class TimedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encode_seconds = 0.
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *arguments, **keywords):
        started = time.perf_counter()
        result = self.tokenizer(*arguments, **keywords)
        self.encode_seconds += time.perf_counter() - started
        self.calls += 1
        return result


def replay_route(arguments, directory, modules, torch):
    from transformers import AutoTokenizer

    started = time.perf_counter()
    rank_paths = [directory / ("rank" + str(rank) + ".json") for rank in (0, 1)]
    rank_reports = [json.loads(path.read_text()) for path in rank_paths]
    gpu_report = rank_reports[0]
    route_key = gpu_report["route_key"]
    report = {
        "schema_version": 1, "status": "RUNNING", "route_key": route_key,
        "run_commit": arguments.run_commit, "probe_run_commit": gpu_report["run_commit"],
        "probe_job_id": gpu_report["job_id"],
        "probe_reports": [record_file(path) for path in rank_paths],
        "cases": [], "model_loaded": False, "gpu_used": False,
    }
    route_path = arguments.output / (route_key + ".json")

    def flush():
        report["elapsed_seconds"] = time.perf_counter() - started
        write(route_path, report)

    try:
        if not re.fullmatch("q(?:06|17)-[LF]", route_key):
            raise AssertionError("Only the four approved Qwen3 routes may be replayed")
        if any(rank_report["route_key"] != route_key or rank_report["run_commit"] != gpu_report["run_commit"] or rank_report["job_id"] != gpu_report["job_id"] for rank_report in rank_reports):
            raise AssertionError("The two GPU rank reports have different route/source/job identities")
        gpu_cases = gpu_report["generation"]["cases"]
        if any(rank_report["generation"]["cases"] != gpu_cases for rank_report in rank_reports):
            raise AssertionError("The gathered bounded generation records differ between GPU rank reports")
        if len(gpu_cases) != 51 or any(case["status"] != "PASS" for case in gpu_cases):
            raise AssertionError("CPU replay requires all51 previously successful bounded beam20 cases")
        if {case["prompt_id"] for case in gpu_cases} != set(range(17)):
            raise AssertionError("The original17 SeqRec templates are incomplete")
        case_map = {(case["prompt_id"], case["row_idx"]): case for case in gpu_cases}
        if len(case_map) != 51:
            raise AssertionError("GPU case identities are not unique")
        samples_path = arguments.samples or Path(gpu_report["samples_identity"]["path"])
        if sha(samples_path) != gpu_report["samples_identity"]["sha256"]:
            raise AssertionError("The CPU sample realization differs from the GPU-bound exact sample document")
        samples = json.loads(samples_path.read_text())
        sample_map = {(case["prompt_id"], case["row_idx"]): case for case in samples["generation"]}
        if set(sample_map) != set(case_map):
            raise AssertionError("Fixed CPU samples and GPU generation identities are not equal")
        report["samples_identity"] = record_file(samples_path)
        tokenizer_directory = arguments.model_tokenizer or Path(gpu_report["save_reload"]["path"])
        tokenizer_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True, trust_remote_code=False)
        report["tokenizer_load_seconds"] = time.perf_counter() - tokenizer_started
        sid_tokens = samples["sid_tokens"]
        if len(sid_tokens) != 930 or tokenizer.convert_tokens_to_ids(sid_tokens) != gpu_report["tokenizer"]["sid_ids"] or len(tokenizer) != gpu_report["tokenizer"]["expanded_tokenizer_size"]:
            raise AssertionError("The saved930-SID Qwen tokenizer does not match the GPU route")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id != gpu_report["tokenizer"]["eos_token_id"]:
            raise AssertionError("The replay tokenizer lost its original PAD/EOS identity")
        if tokenizer.model_max_length != 2048:
            raise AssertionError("The fixed Qwen tokenizer must retain max_length2048")
        report["tokenizer"] = {
            "path": str(tokenizer_directory), "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer), "sid_count": len(sid_tokens),
            "sid_ids_exact": True, "eos_token_id": tokenizer.eos_token_id,
            "files": [record_file(path) for path in sorted(tokenizer_directory.iterdir()) if path.is_file() and path.name in ("tokenizer.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "sid_token_order.json")],
        }
        dataset_arguments = SimpleNamespace(**samples["args_namespace"])
        data_directory = arguments.data.resolve()
        if data_directory.name == "Instruments":
            dataset_arguments.data_path = str(data_directory.parent)
        else:
            dataset_arguments.data_path = str(data_directory)
            data_directory = data_directory / "Instruments"
        if dataset_arguments.dataset != "Instruments" or dataset_arguments.max_his_len != 20:
            raise AssertionError("Only unchanged original Instruments histories of at most20 items are authorized")
        dataset_started = time.perf_counter()
        dataset = modules["data"].SeqRecDataset(dataset_arguments, mode="test", prompt_id=0, sample_num=-1)
        if len(dataset) != 24772 or len(dataset.prompts) != 17 or dataset.indices != samples["indices"]:
            raise AssertionError("Original LC-Rec test data or SID mapping changed")
        all_items = dataset.get_all_items()
        report["dataset_initialization_seconds"] = time.perf_counter() - dataset_started
        report["data"] = {
            "path": str(data_directory), "dataset": "Instruments", "users": len(dataset),
            "max_history_items": dataset_arguments.max_his_len, "his_sep": dataset_arguments.his_sep,
            "add_prefix": dataset_arguments.add_prefix, "unique_canonical_sids": len(all_items),
            "index": record_file(data_directory / ("Instruments" + dataset_arguments.index_file)),
            "interactions": record_file(data_directory / "Instruments.inter.json"),
            "full_test_generation_executed": False,
        }
        timed_tokenizer = TimedTokenizer(tokenizer)
        collator = modules["collator"].TestCollator(dataset_arguments, timed_tokenizer)
        callback_started = time.perf_counter()
        callback = Allowed(tokenizer, dataset.indices)
        callback_seconds = time.perf_counter() - callback_started
        report["callback"] = {
            "initialization_seconds": callback_seconds, "cached_across_cases": True,
            "actual_initializations_this_route": 1,
            "position_sizes": {str(position): len(tokens) for position, tokens in callback.allowed.items()},
            "original_cache_evidence": "Unchanged test_ddp.py builds test_data.get_prefix_allowed_tokens_fn(tokenizer) once before its17-template loop",
            "qwen_adapter": "Reuse the exact Allowed implementation used by the GPU probe; construct legal position sets once and bind the actual encoded prompt per case",
            "runtime_inside_generation": "Already included in real GPU generation_seconds; no beam callback invocations are simulated or charged by this CPU benchmark",
        }
        report["protocol"] = {
            "users_formal": 24772, "templates": 17, "beam": 20, "return_sequences": 20,
            "case_count": 51, "warmup_passes": 1, "measured_repetitions": 3,
            "callback_initializations_formal_center": 1,
            "callback_initializations_formal_conservative": 17,
            "cache_conservative_interpretation": "One initialization per template is a deliberate upper allowance; the original code initializes once for all templates",
            "formal_recall_ndcg_computed": False,
            "filter_items": False, "candidate_deduplication_added": False,
            "cpu_scores_vs_gpu_scores": "Exact saved FP32 values reconstructed as CPU torch.Tensor for the unchanged evaluator; ordering and topk bits must equal the GPU reference. GPU scalar sort/synchronization latency is not measured",
            "phase_overlap": "encode_seconds is inside collator_seconds; parser_seconds is an extra diagnostic parsing check overlapping evaluator semantics and is excluded from total_per_user_seconds",
            "total_scope": "Original Dataset getitem + TestCollator including encoding/tensor packaging + cached Qwen callback bind + CPU sequence/score reconstruction + batch_decode + original get_topk_results including parsing/sorting",
            "excluded": ["Already-measured model.generate including native callback runtime", "One-time dataset/tokenizer/callback initialization", "Diagnostic parser duplicate", "Exact-reference assertions and evidence serialization"],
            "unknown_costs": ["Pinned-memory preparation and H2D transfer", "GPU output D2H and score-sorting synchronization difference", "Original per-batch distributed object collection/barrier and full-result aggregation", "DataLoader worker startup, prefetch and concurrency effects", "Full-run runtime variability and logging"],
        }
        report["original_source"] = {
            name: record_file(arguments.original / (name + ".py"))
            for name in ("data", "collator", "prompt", "evaluate", "test_ddp")
        }
        user_identifiers = list(dataset.inters)

        def replay_case(case):
            source = sample_map[(case["prompt_id"], case["row_idx"])]
            encode_before = timed_tokenizer.encode_seconds
            encode_calls_before = timed_tokenizer.calls
            total_started = time.perf_counter()
            phase_started = time.perf_counter()
            dataset.set_prompt(case["prompt_id"])
            raw = dataset[case["row_idx"]]
            getitem_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            inputs, targets = collator([raw])
            collator_seconds = time.perf_counter() - phase_started
            encode_seconds = timed_tokenizer.encode_seconds - encode_before
            phase_started = time.perf_counter()
            prompt_ids = inputs["input_ids"][0].tolist()
            callback.bind(prompt_ids, raw["input_ids"])
            bind_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            full_sequences = torch.tensor([prompt_ids + suffix for suffix in case["generated_token_ids"]], dtype=torch.long)
            scores = torch.tensor(case["generated_scores"], dtype=torch.float32)
            reconstruction_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            decoded = tokenizer.batch_decode(full_sequences, skip_special_tokens=True)
            decode_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            topk = modules["evaluate"].get_topk_results(decoded, scores, targets, 20, all_items=None)
            evaluator_seconds = time.perf_counter() - phase_started
            total_seconds = time.perf_counter() - total_started
            phase_started = time.perf_counter()
            parsed = [text.split("Response:")[-1].strip().replace(" ", "") for text in decoded]
            parser_seconds = time.perf_counter() - phase_started
            user_ref = text_digest("LC-Rec Instruments user:" + user_identifiers[case["row_idx"]])
            if raw != {"input_ids": source["input_text"], "labels": source["target_sid"]}:
                raise AssertionError("Original Dataset prompt/answer bytes changed during CPU replay")
            if prompt_ids != source["input_ids"] or len(prompt_ids) != case["input_tokens"]:
                raise AssertionError("Original TestCollator Qwen token IDs differ from the fixed GPU prompt")
            if user_ref != source["user_ref"] or user_ref != case["user_ref"] or targets != [case["target_sid"]]:
                raise AssertionError("Original user/target pairing changed")
            if text_digest(raw["input_ids"]) != case["input_text_sha256"]:
                raise AssertionError("Natural-language prompt SHA differs from the GPU input")
            if parsed != case["generated_text"] or topk != case["official_evaluator_topk"]:
                raise AssertionError("CPU full-sequence decode/parser/evaluator output differs from the GPU reference")
            if inputs["attention_mask"].tolist() != [[1] * len(prompt_ids)] or timed_tokenizer.calls - encode_calls_before != 1:
                raise AssertionError("Unexpected batch1 padding or TestCollator encoding count")
            if full_sequences.shape[0] != 20 or scores.shape[0] != 20:
                raise AssertionError("Every CPU replay must retain all20 returned beams")
            return {
                "dataset_getitem_seconds": getitem_seconds, "collator_seconds": collator_seconds,
                "encode_seconds": encode_seconds, "callback_bind_seconds": bind_seconds,
                "cpu_sequence_score_reconstruction_seconds": reconstruction_seconds,
                "decode_seconds": decode_seconds, "parser_seconds": parser_seconds,
                "evaluator_seconds": evaluator_seconds, "total_per_user_seconds": total_seconds,
                "exact_reference_outputs": True,
                "input_tokens": len(prompt_ids), "parsed_sid_sha256": value_digest(parsed),
                "decoded_full_text_sha256": value_digest(decoded), "evaluator_topk": topk,
            }

        ordered_cases = sorted(gpu_cases, key=lambda case: (case["prompt_id"], case["row_idx"]))
        warmup_started = time.perf_counter()
        for case in ordered_cases:
            replay_case(case)
        report["warmup"] = {"cases": 51, "seconds": time.perf_counter() - warmup_started, "all_exact_reference_outputs": True, "included_in_formal_per_user_estimate": False}
        raw_repetitions = {(case["prompt_id"], case["row_idx"]): [] for case in ordered_cases}
        for repetition in range(3):
            repetition_started = time.perf_counter()
            for case in ordered_cases:
                timing = replay_case(case)
                timing["repetition"] = repetition
                raw_repetitions[(case["prompt_id"], case["row_idx"])].append(timing)
            report.setdefault("repetitions", []).append({"repetition": repetition, "cases": 51, "wall_seconds": time.perf_counter() - repetition_started})
            flush()
        for case in ordered_cases:
            timings = raw_repetitions[(case["prompt_id"], case["row_idx"])]
            phase_names = [name for name in timings[0] if name.endswith("_seconds")]
            report["cases"].append({
                "prompt_id": case["prompt_id"], "row_idx": case["row_idx"], "user_ref": case["user_ref"],
                "history_bucket": case["history_bucket"], "status": "PASS",
                "input_tokens": case["input_tokens"], "returned_sequences": 20,
                "total_per_user_seconds": statistics.fmean(timing["total_per_user_seconds"] for timing in timings),
                "conservative_per_user_seconds": max(timing["total_per_user_seconds"] for timing in timings),
                "phase_mean_seconds": {name: statistics.fmean(timing[name] for timing in timings) for name in phase_names},
                "repetitions": timings, "exact_reference_outputs": True,
            })
        if callback.calls or torch.cuda.is_initialized():
            raise AssertionError("CPU postprocessing must neither invoke beam callbacks nor initialize CUDA")
        report["callback"]["beam_callback_invocations_in_cpu_probe"] = 0
        report["cpu_summary"] = {
            "mean_per_user_seconds": statistics.fmean(case["total_per_user_seconds"] for case in report["cases"]),
            "max_per_user_seconds": max(case["conservative_per_user_seconds"] for case in report["cases"]),
            "formal_case_presentations": 24772 * 17,
            "center_two_rank_postprocess_seconds_excluding_one_time_setup": sum(24772 / 2 * statistics.fmean(case["total_per_user_seconds"] for case in report["cases"] if case["prompt_id"] == prompt_id) for prompt_id in range(17)),
            "conservative_two_rank_postprocess_seconds_excluding_one_time_setup": sum(24772 / 2 * max(case["conservative_per_user_seconds"] for case in report["cases"] if case["prompt_id"] == prompt_id) for prompt_id in range(17)),
            "not_a_statistical_confidence_interval": True,
        }
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "PARTIAL" if report.get("cases") else "BLOCKED"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["cuda_initialized"] = torch.cuda.is_initialized()
        flush()
    return report


def main():
    arguments = parse_cli()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "RUNNING", "run_commit": arguments.run_commit,
        "argv": sys.argv, "cwd": str(Path.cwd()), "interpreter": sys.executable,
        "script": record_file(__file__), "common_script": record_file(Path(__file__).with_name("common.py")),
        "started_utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "gpu_used": False, "model_loaded": False, "routes": [],
    }
    try:
        import torch

        torch.set_num_threads(arguments.cpu_threads)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was unexpectedly initialized before the CPU-only benchmark")
        summary["environment"] = {
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
            "cpu_threads": torch.get_num_threads(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
        modules = import_original(arguments.original)
        directories = probe_directories(arguments)
        route_keys = [route_identity(directory) for directory in directories]
        if len(route_keys) != len(set(route_keys)):
            raise AssertionError("Multiple GPU attempts for the same route require explicit --probe selection; output identities cannot be overwritten")
        for directory in directories:
            try:
                report = replay_route(arguments, directory, modules, torch)
            except Exception as error:
                report = {
                    "route_key": route_identity(directory), "status": "BLOCKED", "run_commit": arguments.run_commit,
                    "reason": "Missing or unreadable fixed GPU rank evidence; independent routes continue",
                    "probe_directory": str(directory), "gpu_used": False, "model_loaded": False,
                    "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                }
                write(arguments.output / (report["route_key"] + ".json"), report)
            summary["routes"].append({"route_key": report["route_key"], "status": report["status"], "probe_run_commit": report.get("probe_run_commit"), "probe_job_id": report.get("probe_job_id"), "output": record_file(arguments.output / (report["route_key"] + ".json"))})
            write(arguments.output / "summary.json", summary)
        summary["status"] = "PASS" if all(route["status"] == "PASS" for route in summary["routes"]) else "PARTIAL"
    except Exception as error:
        summary["status"] = "PARTIAL" if summary["routes"] else "BLOCKED"
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(summary["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        summary["finished_utc"] = datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()
        write(arguments.output / "summary.json", summary)
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
