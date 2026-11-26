"""Independently audit the terminal formal run from immutable captured files."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import struct


MODES = ("Normal", "Shuffle3a", "Shuffle4")
STAGES = ("quantizer", "sid", "bound", "training", *MODES, "summary")
USERS = 24772
ITEMS = 9922
DISCOUNTS = tuple(1 / math.log2(rank + 1) for rank in range(1, 11))
FP32_TOLERANCE = 24 * 2**-24


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def verify_terminal(job, status, launch):
    request = load(job / "request.json")
    assert request == load(launch / "request.json")
    digest = hashlib.sha256(canonical(request)).hexdigest()
    raw = load(job / "status.json")
    for key in ("job_id", "status", "exit_code", "started_at", "ended_at", "worker", "supervisor", "cleanup", "artifacts"):
        assert status[key] == raw[key], key
    for filename in ("accepted.json", "attempt.json"):
        assert load(job / filename) == load(launch / filename)
        assert load(job / filename)["request_sha256"] == digest
    start = load(launch / "start-query.json")
    for key in ("started_at", "worker", "supervisor", "job_id"):
        assert status[key] == start[key], key
    assert status["request_sha256"] == digest
    assert status["exit_code"] == 0 and status["status"] == "SUCCEEDED"
    assert not (status["worker_alive"] or status["supervisor_alive"] or status["workgroup_live"])
    assert status["cleanup_verified"] and status["workgroup_observation"]["verified_empty"]
    assert load(job / "cleanup.json") == status["cleanup"]
    assert status["cleanup"]["verified"] and status["cleanup"]["reason"] == "NORMAL_EXIT_CHECK"
    assert len(status["artifacts"]) == len({row["path"] for row in status["artifacts"]})
    launch_manifest = load(launch / "closed/capture.json")["files"]
    paths = {}
    preserved_launch = 0
    for record in status["artifacts"]:
        name = record["path"]
        if name in launch_manifest:
            old = launch_manifest[name]
            assert (record["bytes"], record["sha256"]) == (old["bytes"], old["sha256"]), name
            preserved_launch += 1
        for candidate in (job / name, launch / "closed" / name):
            if candidate.exists():
                assert not candidate.is_symlink(), candidate
                assert (candidate.stat().st_size, sha(candidate)) == (record["bytes"], record["sha256"]), candidate
                paths[name] = candidate
                break
    run = job / "output/run"
    startup = load(paths["output/run/startup.json"])
    assert startup["version"] == request["capture_version"] and startup["scope"] == request["scope"]
    assert startup["job"] == request["job_id"] and startup["mode"] == "formal"
    stages = {}
    file_index = {row["path"]: row["sha256"] for row in status["artifacts"]}
    for stage in STAGES:
        receipt = load(paths[f"output/run/stages/{stage}.json"])
        assert receipt["stage"] == stage and receipt["mode"] == "formal"
        assert receipt["exit_code"] == 0 and receipt["reused_from"] is None
        folder = {
            "quantizer": "upstream", "sid": "upstream", "bound": "bound", "training": "training",
            "summary": "evaluation",
        }.get(stage, "evaluation/" + stage)
        for name, digest_ in receipt["artifacts"].items():
            assert file_index[f"output/run/{folder}/{name}"] == digest_, (stage, name)
        stages[stage] = {"seconds": receipt["seconds"], "files": len(receipt["artifacts"]),
                         "argv": receipt["argv"]}
    log = (job / "work.log").read_text()
    names = [line.split()[1] for line in log.splitlines() if line.startswith("Starting ") and line.endswith(" (formal)")]
    assert names == list(STAGES)
    assert "Traceback (most recent call last)" not in log and "CUDA out of memory" not in log
    missing = sorted(set(file_index) - paths.keys())
    extracted_path = launch / "quantizer-checkpoint/checkpoint_extraction.json"
    prior_quantizer = load(extracted_path)
    absent_quantizer = "output/run/upstream/quantizer_train/checkpoints/" + Path(prior_quantizer["source"]).name
    assert not set(missing) - {absent_quantizer}, missing
    assert file_index[absent_quantizer] == prior_quantizer["checkpoint_sha256"]
    assert (load(paths["output/run/upstream/fit_gate.json"])["checkpoint_sha256"] ==
            prior_quantizer["checkpoint_sha256"])
    checkpoints = [row for row in status["artifacts"] if row["path"].startswith("output/run/training/checkpoints/")]
    (checkpoint,) = checkpoints
    best = load(run / "training/best.json")
    assert checkpoint["path"] == "output/run/training/" + best["file"]
    assert checkpoint["sha256"] == best["sha256"]
    return request, run, log, {"stages": stages, "runner_artifacts": len(file_index),
       "rehashed_artifacts": len(paths), "unchanged_launch_artifacts": preserved_launch,
       "runner_artifacts_without_raw_checkpoint_copy": missing, "request_sha256": digest,
       "started_at": status["started_at"], "ended_at": status["ended_at"]}


def verify_training(run, log, checkpoint, launch):
    folder = run / "training"
    training = load(folder / "training.json")
    best = load(folder / "best.json")
    model = load(folder / "model.json")
    assert model == load(launch / "cutoff/output/run/training/model.json")
    with (folder / "validation.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    assert rows == training["validation"]
    assert len(rows) > 10 and all(row["batches"] == 3097 for row in rows)
    assert [row["step"] for row in rows] == list(range(100, 100 * (len(rows) + 1), 100))
    assert training["updates"] == rows[-1]["step"]
    assert training["optimizer_evidence"] == [] and training["legacy_shuffle_calls"] == 0
    assert training["train_seed"] == 42
    assert sum(training["expanded_microbatch"].values()) == 16 * training["updates"]
    assert all(0 < int(size) <= 128 and count > 0 for size, count in training["expanded_microbatch"].items())
    metrics = csv_rows(folder / "csv/version_0/metrics.csv")
    assert all(math.isfinite(float(value)) for row in metrics for value in row.values() if value != "")
    validation_csv = [row for row in metrics if row.get("val/recall@5")]
    for callback_row, csv_row in zip(rows, validation_csv, strict=True):
        assert callback_row["recall_at_5"] == float(csv_row["val/recall@5"])
    top = max(rows, key=lambda row: row["recall_at_5"])
    assert best["best_step"] == top["step"] and best["score"] == top["recall_at_5"]
    assert best["mode"] == "formal"
    assert best["sid_sha256"] == load(launch / "closed/output/run/upstream/sid_gate.json")["sid_sha256"]
    assert best["binding_sha256"] == sha(launch / "closed/output/run/bound/binding.json")
    patience_count = (rows[-1]["step"] - top["step"]) // 100
    assert patience_count == 10 and "did not improve in the last 10 records" in log
    extraction = load(checkpoint)
    assert extraction["source"].endswith("/training/" + best["file"])
    assert extraction["checkpoint_sha256"] == best["sha256"]
    assert extraction["state_sha256"] == best["state_sha256"]
    assert extraction["global_step"] == best["best_step"]
    callback = extraction["selection"]
    assert callback["monitor"] == "val/recall@5" and "'mode': 'max'" in callback["key"]
    assert callback["best_model_path"] == extraction["source"]
    assert callback["best_model_score"] == best["score"] and callback["best_k_models"] == {
        extraction["source"]: best["score"]}
    early = extraction["early_stopping_at_best"]
    assert early["patience"] == patience_count and early["best_score"] == best["score"] and early["wait_count"] == 0
    optimizer = extraction["optimizer"]
    assert optimizer["state_entries"] == model["dtype_counts"]["torch.float32"] == 127
    assert optimizer["step_values"] == [best["best_step"]]
    assert all(group["lr"] == 0.001 and group["weight_decay"] == 0.0001 for group in optimizer["param_groups"])
    assert len(list((folder / "checkpoints").glob("*.ckpt"))) <= 1
    if (folder / best["file"]).is_file():
        assert sha(folder / best["file"]) == best["sha256"]
    assert len(optimizer["param_groups"]) == 1
    return best, {"updates": training["updates"], "training_microbatches": sum(training["expanded_microbatch"].values()),
                   "complete_validations": len(rows), "validation_users_each": USERS,
                   "best": best, "last_validation": rows[-1], "last_validation_csv": validation_csv[-1],
                   "patience_count": patience_count, "training_seconds": training["seconds"],
                   "best_checkpoint_state_keys": extraction["state_keys"], "best_checkpoint_adam": optimizer}


def shuffle_choices(width, row, length, projection):
    catalog = tuple(permutations(range(width)))
    limit = (1 << 256) - (1 << 256) % len(catalog)
    tails = tuple(range(width, 4))
    draws = []
    for occurrence in range(length):
        for counter in range(100):
            key = f'["GRID-LETTER-SHUFFLE-V2",42,{width},{row},{occurrence},{counter}]'
            value = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            if value < limit:
                index = value % len(catalog)
                break
        else:
            raise ValueError("Unbounded rejection during independent replay")
        assert index == projection[occurrence]
        draws.append([list(catalog[index] + tails), index, counter])
    return draws


def metric_row(record, sizes):
    score = struct.unpack("<10f", bytes.fromhex(record["scores_f32le"]))
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in score)
    sids = list(map(tuple, record["ids"]))
    assert len(sids) == 10 and all(len(sid) == 4 and all(type(v) is int and 0 <= v < 256 for v in sid) for sid in sids)
    target = tuple(record["target_sid"])
    ranks = record["native_topk"]
    scores_descending = sorted(score, reverse=True)
    stable = sorted(range(10), key=lambda i: (-score[i], i))
    flags = [sid == target for sid in sids]
    native = {}
    assert set(ranks) == {f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (5, 10)}
    for kind, indices in ranks.items():
        k = int(kind.split("@")[1])
        assert len(indices) == len(set(indices)) == k and all(0 <= i < 10 for i in indices)
        assert [score[i] for i in indices] == scores_descending[:k]
        places = [rank for rank, index in enumerate(indices) if flags[index]]
        relevant = min(k, sum(flags))
        if relevant and kind.startswith("ndcg"):
            native[kind] = math.fsum(DISCOUNTS[place] for place in places) / math.fsum(DISCOUNTS[:relevant])
        else:
            native[kind] = len(places) / relevant if relevant else 0.0
    hits = [rank for rank, index in enumerate(stable, 1) if flags[index]]
    first = hits[0] if hits else 0
    unified = {"hit@1": float(first == 1)}
    before_target = sum(sizes.get(sids[i], 1) for i in stable[:first - 1]) if first else 0
    group = sizes[target]
    for k in (5, 10):
        qualified = 0 < first <= k
        unified[f"hit@{k}"] = float(qualified)
        unified[f"mrr@{k}"] = 1 / first if qualified else 0.0
        unified[f"ndcg@{k}"] = math.fsum(DISCOUNTS[r - 1] for r in hits if r <= k)
        available = min(group, max(0, k - before_target)) if first else 0
        unified[f"ItemHit@{k}"] = available / group
        unified[f"ItemNDCG@{k}"] = sum(DISCOUNTS[before_target:before_target + available]) / group
    counts = {"candidates": 10, "out_of_catalog": sum(sid not in sizes for sid in sids),
              "duplicate_sids": 10 - len(set(sids)), "tied_score_rows": len(set(score)) != 10,
              "native_stable_order_differences": any(idx != stable[:len(idx)] for idx in ranks.values()),
              "collision_target_rows": group > 1}
    return {"unified_order": stable, "first_sid_rank": first, "native": native, **unified}, counts


def check_evaluations(run, launch, request, best, state_keys):
    data_root = launch / "closed/inputs"
    for asset in ("Instruments.inter.json", "user_axis.json", "prior_v2_replay_projection.jsonl"):
        assert sha(data_root / asset) == request["scope"]["input_hashes"][asset]
    users = load(data_root / "Instruments.inter.json")
    axis_data = load(data_root / "user_axis.json")
    axis = axis_data.get("user_to_global_row_idx", axis_data)
    sid_table = launch / "closed/output/run/upstream/item_sids.jsonl"
    with sid_table.open() as stream:
        table = [json.loads(line) for line in stream]
    assert [row["item_id"] for row in table] == list(range(ITEMS))
    codes = [row["sid"] for row in table]
    sizes = Counter(map(tuple, codes))
    assert len(sizes) == ITEMS and set(sizes.values()) == {1}
    stats = {mode: Counter() for mode in MODES}
    sweeps = {mode: Counter() for mode in MODES}
    transitions = {mode: Counter() for mode in MODES[1:]}
    draws_seen = {mode: Counter() for mode in MODES}
    distributions = {mode: defaultdict(list) for mode in MODES}
    eval_root = run / "evaluation"
    with ExitStack() as stack:
        inputs = [stack.enter_context((eval_root / mode / "candidates.jsonl").open()) for mode in MODES]
        metrics = [stack.enter_context((eval_root / mode / "per_user_metrics.jsonl").open()) for mode in MODES]
        projection_stream = stack.enter_context((data_root / "prior_v2_replay_projection.jsonl").open())
        for uid, value in enumerate(zip(*inputs, *metrics, projection_stream, strict=True)):
            assert uid < USERS
            records = list(map(json.loads, value[:3]))
            per_user = list(map(json.loads, value[3:6]))
            projection = json.loads(value[6])
            user = str(uid)
            sequence = users[user]
            row = axis[user]
            history, target = sequence[-21:-1], sequence[-1]
            assert (projection["user_id"], projection["row_idx"], projection["target_item_id"]) == (user, row, target)
            assert projection["visible_letter_ids"] == history and row == uid
            history_codes = [v for item in history for v in codes[item]]
            padding = 120 - len(history_codes)
            before = [code + 256 * (i % 4) for i, code in enumerate(history_codes)]
            for mode, record, ledger in zip(MODES, records, per_user, strict=True):
                assert (record["user_id"], record["global_row"], record["mode"]) == (user, row, mode)
                assert record["target_item_id"] == target and record["target_sid"] == codes[target]
                assert record["history_item_ids"] == history
                assert record["input_ids"] == history_codes + [-1] * padding
                assert record["input_mask"] == [1] * len(history_codes) + [0] * padding
                assert record["offset_ids_before"] == before + [0] * padding
                choices = ([[[0, 1, 2, 3], 0, 0]] * len(history) if mode == "Normal" else
                           shuffle_choices(3 if mode == "Shuffle3a" else 4, row, len(history),
                                           projection["a0_and_a1_permutation_ids"][mode.lower()]))
                assert record["draws"] == choices
                after = []
                for start, (perm, index, rejected) in enumerate(choices):
                    group = before[4 * start:4 * start + 4]
                    moved = [group[i] for i in perm]
                    assert sorted(moved) == group
                    assert mode != "Shuffle3a" or moved[3] == group[3]
                    sweeps[mode].update(history_groups=1, nonidentity_draws=index != 0, changed_token_groups=moved != group)
                    draws_seen[mode][index] += 1
                    after.extend(moved)
                assert record["offset_ids_after"] == after + [0] * padding
                if mode != "Normal":
                    transitions[mode].update(candidate_id_rows_changed=record["ids"] != records[0]["ids"],
                                             candidate_score_rows_changed=record["scores_f32le"] != records[0]["scores_f32le"])
                independent, counts = metric_row(record, sizes)
                assert ledger == {"global_row": row, "user_id": user, **independent}, (mode, user)
                stats[mode].update(counts)
                distributions[mode]["n_users"].append(1)
                for k, val in independent["native"].items():
                    distributions[mode][("native", k)].append(val)
                for k, val in independent.items():
                    if "@" in k:
                        distributions[mode][("unified", k)].append(val)
    proof, all_metrics = {}, {}
    for mode in MODES:
        assert len(distributions[mode].pop("n_users")) == USERS
        metrics_out = {"native": {}, "unified": {}}
        for (group, name), values in distributions[mode].items():
            metrics_out[group][name] = math.fsum(values) / USERS
        original_offline = load(eval_root / mode / "offline_evaluation.json")
        original_native = load(eval_root / mode / "native_evaluation.json")
        assert original_offline["coverage"] == USERS and stats[mode]["candidates"] == USERS * 10
        assert original_offline["counts"] == dict(stats[mode])
        assert original_offline["metrics"] == metrics_out
        assert sweeps[mode]["history_groups"] == 172929
        assert original_native["checks"] == dict(sweeps[mode]) | {"batch_size_8": 3096, "batch_size_4": 1}
        assert original_native["eval_seed"] == 2026 and original_native["sid_sha256"] == best["sid_sha256"]
        assert original_native["reload"] == {"strict": True, "keys": state_keys, "checkpoint_sha256": best["sha256"],
                                               "state_sha256": best["state_sha256"]}
        assert original_native["candidate_sha256"] == sha(eval_root / mode / "candidates.jsonl")
        delta = {name: abs(value - original_native["metrics"]["test/" + name])
                 for name, value in metrics_out["native"].items()}
        assert original_offline["native_abs_delta"] == delta and max(delta.values()) <= FP32_TOLERANCE
        (native_csv,) = csv_rows(eval_root / mode / "csv/version_0/metrics.csv")
        for key, val in original_native["metrics"].items():
            assert val == float(native_csv[key])
        proof[mode] = {"all_per_user_ledgers_equal": USERS, "native_abs_delta": delta, "counts": dict(stats[mode]),
                       "actual_draw_indices": dict(draws_seen[mode]), "actual_intervention": dict(sweeps[mode]),
                       "metrics": metrics_out, "native_metrics": original_native["metrics"],
                       "total_seconds": original_native["seconds"]}
        all_metrics[mode] = metrics_out
    parent_summary = load(eval_root / "summary.json")
    assert parent_summary["mode"] == "formal" and parent_summary["best_sha256"] == best["sha256"]
    delta = {}
    for mode in MODES:
        assert parent_summary["candidate_metrics"][mode] == load(eval_root / mode / "offline_evaluation.json")
        if mode == "Normal":
            continue
        groups = {}
        for group, metrics_out in all_metrics[mode].items():
            groups[group] = {}
            for name, value in metrics_out.items():
                base = all_metrics["Normal"][group][name]
                difference = value - base
                groups[group][name] = {"absolute": difference, "relative": difference / base if base else None}
        comparisons = dict(transitions[mode])
        assert parent_summary["relative_to_normal"][mode] == {"metrics": groups, **comparisons,
                **dict(sweeps[mode]), "batch_size_8": 3096, "batch_size_4": 1}
        delta[mode] = {"metrics": groups, **comparisons}
    return {"conditions": proof, "relative_to_normal": delta}


def verify_configuration(run, launch):
    import yaml

    def digest_path(folder):
        return folder / "effective_config.yaml"

    def parse(folder):
        return yaml.safe_load(digest_path(folder).read_bytes())

    def relocate(value, old, new):
        if isinstance(value, str):
            return value.replace(old, new)
        if isinstance(value, list):
            return [relocate(item, old, new) for item in value]
        if isinstance(value, dict):
            return {name: relocate(item, old, new) for name, item in value.items()}
        return value

    training = parse(run / "training")
    assert training == parse(launch / "cutoff/output/run/training")
    trainer = training["trainer"]
    assert trainer["accumulate_grad_batches"] == 16 and trainer["val_check_interval"] == 1600
    assert trainer["precision"] == "32-true" and trainer["devices"] == 1 and trainer["deterministic"]
    callbacks = training["callbacks"]
    assert callbacks["model_checkpoint"]["mode"] == callbacks["early_stopping"]["mode"] == "max"
    assert callbacks["model_checkpoint"]["save_top_k"] == 1 and callbacks["early_stopping"]["patience"] == 10
    assert training["model"]["eval_seed"] == 2026 and training["model"]["top_k_for_generation"] == 10
    assert not training["model"]["shuffle_embedding_on_test"]
    assert training["data_loading"]["test_dataloader_config"]["dataloader"]["batch_size_per_device"] == 8
    output = {"training": sha(digest_path(run / "training"))}
    for mode in MODES:
        observed = parse(run / "evaluation" / mode)
        expected = relocate(training, training["paths"]["output_dir"], observed["paths"]["output_dir"])
        expected["id"] = observed["id"]
        assert observed == expected, (mode, [key for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key)])
        output[mode] = sha(digest_path(run / "evaluation" / mode))
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--original-query", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    request, run, log, terminal = verify_terminal(args.job, load(args.original_query), args.launch)
    best, training = verify_training(run, log, args.checkpoint, args.launch)
    config = verify_configuration(run, args.launch)
    metrics = check_evaluations(run, args.launch, request, best, training["best_checkpoint_state_keys"])
    receipt = {"terminal": terminal, "training": training, "evaluation": metrics, "effective_config_hashes": config,
               "verified_at": datetime.now(timezone.utc).isoformat()}
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
