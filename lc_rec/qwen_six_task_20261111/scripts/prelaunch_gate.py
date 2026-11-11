"""Check exact CPU identities and explain the initial safe micro layouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as numpy_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    workloads = {}
    samples = {}
    for key in ("q06", "q17"):
        folder = args.evidence / "jobs" / ("lc-qwen6-20261111-cpu-" + key + "-r1") / "output/audit"
        report = json.loads((folder / "audit.json").read_text())
        assert report["status"] == "PASS_CPU_WITH_FORMAL_DATA_BLOCKER" and report["exit_code"] == 0
        assert report["concat_rows"] == 221225
        assert report["tokenizer"]["added_sid_tokens"] == 930
        assert report["tokenizer"]["expanded_tokenizer_size"] == 152599
        assert report["mask"]["checked_records"] == 221225
        assert report["mask"]["original_natural_language_text_hashes_exact"]
        assert report["mask"]["cross_entropy_temperature"] == 1.
        samples[key] = json.loads((folder / "samples.json").read_text())
        with numpy_module.load(folder / "workload.npz", allow_pickle=False) as archive:
            workloads[key] = {name: archive[name].copy() for name in archive.files}
        reports[key] = report
    assert workloads["q06"].keys() == workloads["q17"].keys()
    assert all(numpy_module.array_equal(workloads["q06"][name], workloads["q17"][name]) for name in workloads["q06"])
    for name in ("mixed_global128_0", "mixed_global128_1"):
        assert samples["q06"]["plans"][name] == samples["q17"]["plans"][name]
    routes = {}
    for key, route, micro in (("q06", "L", 2), ("q06", "F", 2), ("q17", "L", 2), ("q17", "F", 1)):
        report = reports[key]
        longest = max(task["full_tokens"]["max"] for task in report["per_task"].values())
        persistent = report["memory_static"]["lora_zero2_persistent_bytes_per_gpu" if route == "L" else "full_zero2_persistent_bytes_per_gpu"]
        logits = int(longest) * micro * report["memory_static"]["logits_float32_bytes_per_micro_and_token"]
        routes[key + "-" + route] = {
            "micro": micro, "accumulation": 128 // (2 * micro), "global_batch": 128,
            "observed_qwen_longest_tokens": longest, "estimated_persistent_gib_per_rank": persistent / 1024 ** 3,
            "one_float32_logits_tensor_at_longest_gib": logits / 1024 ** 3,
            "layout_basis": "Conservative initial capacity choice from actual1513-token tail and unique-parameter ZeRO2 state accounting; Q17 full uses micro1 to leave room for CE temporaries, gradients, communication and allocator reserve",
            "estimate_is_not_gpu_pass": True, "other_micro_layouts": "NOT TESTED; this task does not claim a fastest layout",
        }
    report = {"status": "PASS_CPU_IDENTITY_FOR_BOUNDED_COMPUTE_WITH_SEMANTIC_LIMITATION",
              "both_models_same_original_records_prompts_and_all_measured_arrays": True,
              "model_payload_downloads": 0, "routes": routes,
              "mask_boundary": {
                  "mismatch_records": reports["q06"]["mask"]["boundary_mismatch_count"],
                  "rule": "Retain original independently tokenized prompt-length masking; only restore genuine EOS using attention_mask when Qwen PAD==EOS",
                  "observed_limitation": "Qwen may merge the final prompt colon with the first answer characters; retaining the original mask can omit those answer characters and turn additional short answers into EOS-only supervision",
                  "not_claimed": "Complete natural-language answer supervision or exact LLaMA numerical supervision",
                  "offset_mask_or_separate_segment_retokenization": "NOT APPLIED; changing the actual supervised token range is an important method decision outside this unchanged-mask technical probe",
                  "computation_scope": "Record original mask behavior truthfully, measure all six tasks and fixed assets, and keep semantic-fidelity status separate from compute PASS",
              }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "routes": routes, "mask_boundary_mismatch": report["mask_boundary"]["mismatch_records"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
