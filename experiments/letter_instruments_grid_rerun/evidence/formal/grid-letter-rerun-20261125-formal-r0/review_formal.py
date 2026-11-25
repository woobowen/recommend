"""Verify the staged formal run evidence without starting any model process."""

from argparse import ArgumentParser
from collections import Counter
import csv
from datetime import datetime
from gzip import open as gzip_open
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
from zipfile import ZipFile


def load(path):
    return json.loads(path.read_bytes())


def load_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def verify_launch(root, source, runner):
    request = load(root / "request.json")
    accepted = load(root / "accepted.json")
    attempt = load(root / "attempt.json")
    query = load(root / "start-query.json")
    submit = load(root / "submit-response.json")
    confirm = load(root / "start-confirm.json")
    preflight = load(root / "remote-preflight.json")
    digest = sha256(compact(request)).hexdigest()
    assert all(record["request_sha256"] == digest for record in (accepted, attempt, query, submit))
    assert confirm["records"] == {"request": request, "accepted": accepted, "attempt": attempt}
    assert not preflight["job_exists"] and request["job_id"] not in preflight["jobs"]
    assert request["job_id"] + ".json" not in preflight["saved_requests"]
    assert submit["submission"] == "ACCEPTED" and query["status"] == "RUNNING"
    assert query["worker_alive"] and query["supervisor_alive"]
    assert query["started_at"] == confirm["status"]["started_at"]
    assert request["capture_version"] == preflight["version"] == source.joinpath("VERSION").read_text().strip()
    assert accepted["runner_sha256"] == preflight["runner_sha256"] == sha(runner)
    assert request["deployment_sha256"] == preflight["deployment_sha256"]
    assert request["timeout_seconds"] == 86400 and request["scope"]["reuse_job"] is None
    assert request["argv"][-3:] == ["execute", "--mode", "formal"]
    scope = request["scope"]
    actual = {file: sha(source / file) for file in scope["source_hashes"]}
    assert actual == scope["source_hashes"]
    base = Path(request["cwd"]).parent
    expected = {str(Path(request["cwd"]) / name): hash_ for name, hash_ in actual.items()}
    expected.update({str(base / "inputs" / name): hash_ for name, hash_ in scope["input_hashes"].items()})
    assert len(request["inputs"]) == len(expected)
    assert {rec["path"]: rec["sha256"] for rec in request["inputs"]} == expected

    worker = confirm["processes"]["worker"]
    supervisor = confirm["processes"]["supervisor"]
    assert worker["argv"] == request["argv"] and worker["cwd"] == request["cwd"]
    assert worker["ppid"] == supervisor["pid"] and supervisor["ppid"] == 1
    for kind, process in (("worker", worker), ("supervisor", supervisor)):
        assert process["pid"] == process["sid"] == process["pgid"] == query[kind]["pid"]
        assert process["start_ticks"] == query[kind]["start_ticks"]
        assert process["tty_nr"] == 0 and process["fds"]["0"] == "/dev/null"
    log_path = str(Path(query["result_path"]).parent / "work.log")
    assert worker["fds"]["1"] == worker["fds"]["2"] == log_path
    gpu = request["env"]["CUDA_VISIBLE_DEVICES"]
    assert gpu == worker["cuda_visible_devices"]
    gpu_pid = next(int(row[1]) for row in csv.reader(confirm["gpu_apps"].splitlines()) if row[0] == gpu)
    assert gpu_pid in {member["pid"] for member in query["workgroup_observation"]["owned_live"]}
    assert confirm["work.log"]["tail"] == "Starting quantizer (formal)\n"

    return request, {
        "canonical_request_sha256": digest, "snapshot_files_verified": len(actual),
        "source_commit": request["capture_version"], "input_scope_entries": len(scope["input_hashes"]),
        "started_at": query["started_at"], "independent_connection_confirmed_at": confirm["time"],
        "supervisor_pid": supervisor["pid"], "worker_pid": worker["pid"], "initial_gpu_pid": gpu_pid,
        "gpu_uuid": gpu, "timeout_seconds": request["timeout_seconds"],
    }


def verify_receipts(run, request):
    output = {}
    remote_root = Path(request["cwd"])
    for stage, folder, name, action in (
        ("quantizer", "upstream", "upstream.py", "fit"),
        ("sid", "upstream", "upstream.py", "sid"),
        ("bound", "bound", "downstream.py", "prepare"),
    ):
        receipt = load(run / "stages" / (stage + ".json"))
        assert receipt["stage"] == stage and receipt["mode"] == "formal"
        assert receipt["exit_code"] == 0 and receipt["reused_from"] is None
        assert receipt["argv"][:6] == [request["argv"][0], "-B", str(remote_root /
            "experiments/letter_instruments_grid_rerun" / name), action, "--mode", "formal"]
        compared, retained_only_on_server = 0, []
        for file, digest in receipt["artifacts"].items():
            path = run / folder / file
            if path.is_file():
                assert sha(path) == digest, path
                compared += 1
            else:
                retained_only_on_server.append(file)
        output[stage] = {"seconds": receipt["seconds"], "files_verified": compared,
                         "receipt_paths_not_replayed": retained_only_on_server}
    return output


def verify_fit(upstream, request):
    fit = load(upstream / "fit_gate.json")
    inputs = load(upstream / "upstream_inputs.json")
    fit_root = upstream / "quantizer_train"
    result = load(fit_root / "stage_result.json")
    checkpoint = load(fit_root / "checkpoint_evidence.json")
    observations = load_rows(fit_root / "fit_batches.jsonl")
    expected_hashes = {name.removesuffix(".pt").removesuffix(".jsonl") + "_sha256":
                       request["scope"]["input_hashes"][name]
                       for name in ("item_embeddings.pt", "item_identity.jsonl", "items_qwen.jsonl")}
    assert fit["input_identity"] == expected_hashes == {key: inputs[key] for key in expected_hashes}
    assert fit["mode"] == inputs["mode"] == "formal"
    assert fit["step"] == result["global_step"] == checkpoint["global_step"] == 30
    assert checkpoint["current_layer"] == 2
    assert checkpoint["layers_initialized"] == [True] * 3
    assert checkpoint["centroid_shapes"] == [[256, 3584]] * 3
    assert checkpoint["centroids_finite"] == [True] * 3
    assert inputs["embedding"] == {"all_finite": True, "dtype": "torch.float32", "rows_verified": 9922,
                                   "shape": [9922, 3584], "tensor_order": "ascending internal item_id"}
    ids = set()
    updates = Counter()
    changed = Counter()
    buffered = Counter()
    initialized = Counter()
    step = 0
    for index, row in enumerate(observations):
        ids.update(row["item_ids"])
        layer = row["layer"]
        assert row["batch_idx"] == index and row["batch_size"] == len(row["item_ids"]) == 2048
        assert math.isfinite(row["loss"]) and all(row["centroids_finite"])
        assert all(not flag or i == layer for i, flag in enumerate(row["centroids_changed"]))
        assert row["step_before"] == step
        if row["step_after"] == step + 1:
            assert row["initialized_before"] == bool(updates[layer])
            assert row["initialized_after"] and not row["initializing"]
            updates[layer] += 1
            changed[layer] += row["centroids_changed"][layer]
            step += 1
        else:
            assert row["step_after"] == step and not row["initialized_before"]
            if row["initializing"]:
                assert not row["initialized_after"] and row["centroids_changed"][layer]
                assert (row["initialization_buffer_before"], row["initialization_buffer_after"]) == (2048, 0)
                initialized[layer] += 1
            else:
                assert not row["initialized_after"] and not any(row["centroids_changed"])
                assert (row["initialization_buffer_before"], row["initialization_buffer_after"]) == (0, 2048)
                buffered[layer] += 1
    assert step == 30 and ids == set(range(9922))
    assert len(observations) == fit["raw_batches"] == 36
    assert fit["rows_processed_with_repeats"] == 73728 and fit["distinct_rows_observed"] == 9922
    assert updates == {layer: 10 for layer in range(3)}
    assert initialized == buffered == {layer: 1 for layer in range(3)}
    for layer, rec in fit["per_layer"].items():
        idx = int(layer)
        assert rec == {"initialization_batches": initialized[idx] + buffered[idx], "initializations": initialized[idx],
                       "updates": updates[idx], "changed_updates": changed[idx]}
        assert changed[idx] > 0
    return {"fit_raw_batches": len(observations), "fit_rows_with_repeats": 73728, "distinct_rows": len(ids),
            "lightning_updates": step, "per_layer_changed_updates": dict(changed),
            "initialization_batches_per_layer": 2, "checkpoint_sha256_from_receipt": fit["checkpoint_sha256"]}


def verify_upstream_catalogue(path, identities, offline):
    with gzip_open(path, "rb") as data:
        for index, identity in enumerate(identities):
            width_bytes = data.read(8)
            (width,) = struct.unpack("<Q", width_bytes)
            assert struct.unpack("<I", data.read(4))[0] == offline.masked_crc(width_bytes)
            assert width <= 100_000
            frame = data.read(width)
            assert len(frame) == width and struct.unpack("<I", data.read(4))[0] == offline.masked_crc(frame)
            values = {}
            for tag, feature in offline.message_fields(offline.only(frame, 1)):
                assert tag == 1
                values_ = dict(offline.message_fields(feature))
                values[values_[1]] = values_[2]
            assert set(values) == {b"id", b"text"}
            assert list(offline.varints(offline.only(offline.only(values[b"id"], 3), 1))) == [index]
            text_bytes = offline.only(offline.only(values[b"text"], 1), 1)
            assert sha256(text_bytes).hexdigest() == identity["text_sha256"]
        assert not data.read(1)


def verify_catalogue(run, common, request, offline):
    upstream = run / "upstream"
    fit = load(upstream / "fit_gate.json")
    sid = load(upstream / "sid_gate.json")
    binding = load(run / "bound/binding.json")
    identifiers = common / "item_identity.jsonl"
    users_path = common / "Instruments.inter.json"
    user_axis_path = common / "user_axis.json"
    provided_inputs = request["scope"]["input_hashes"]
    for path in (identifiers, users_path, user_axis_path):
        assert sha(path) == provided_inputs[path.name]
    identities = load_rows(identifiers)
    user_axis_source = load(user_axis_path)
    user_axis = user_axis_source.get("user_to_global_row_idx", user_axis_source)
    assert len(user_axis) == 24772 and all(user_axis[str(i)] == i for i in range(24772))
    codes, proof = offline.canonical_catalog(run)
    sids = load_rows(upstream / "item_sids.jsonl")
    for pos, (original, assigned) in enumerate(zip(identities, sids, strict=True)):
        assert original["item_id"] == original["letter_item_id"] == assigned["item_id"] == pos
        assert assigned["letter_item_id"] == pos and assigned["asin"] == original["asin"]
    assert len({row["asin"] for row in identities}) == len({row["old_grid_embedding_row"] for row in identities}) == 9922
    item_record = upstream / "catalogue/items/part-00000.tfrecord.gz"
    assert sha(item_record) == load(upstream / "upstream_inputs.json")["catalogue_tfrecord_sha256"]
    verify_upstream_catalogue(item_record, identities, offline)
    prediction = load(upstream / "sid_generation/prediction_observation.json")
    assert prediction == sid["prediction"]
    assert prediction["items_observed"] == sum(prediction["native_batch_sizes"]) == 9922
    assert sid["mode"] == binding["mode"] == "formal"
    assert sid["fit_gate_sha256"] == sha(upstream / "fit_gate.json")
    assert sid["checkpoint_sha256"] == fit["checkpoint_sha256"]
    assert sid["input_identity"] == fit["input_identity"]
    assert sid["predicted_codes_sha256"] == sha(upstream / "sid_generation/predicted_codes_by_item.pt")
    assert sid["item_sids_sha256"] == sha(upstream / "item_sids.jsonl")
    prefixes = Counter(tuple(code[:3]) for code in codes)
    histogram = Counter(prefixes.values())
    assert sid["unique_prefixes"] == proof["prefixes"] == len(prefixes)
    assert sid["unique_sids"] == proof["unique_full_codes"] == 9922
    assert sid["prefix_group_size_histogram"] == {str(k): n for k, n in histogram.items()}
    assert sid["items_in_colliding_prefixes"] == sum(n for n in prefixes.values() if n > 1)
    assert sid["digit_ranges"] == [{"min": min(c), "max": max(c), "used": len(set(c))} for c in zip(*codes)]

    (prepared_sid,) = list((run / "bound").rglob("semantic_ids.pt"))
    prepared = prepared_sid.parent
    catalog = prepared / "catalog.json"
    assert load(catalog) == [{"item": i, "sid": sid} for i, sid in enumerate(codes)]
    assert sha(catalog) == binding["catalog_sha256"]
    assert sid["sid_sha256"] == proof["sha256"] == binding["sid_sha256"] == sha(prepared_sid)
    assert binding["gate_sha256"] == sha(upstream / "sid_gate.json")
    assert binding["identity_sha256"] == sha(identifiers)
    assert binding["interactions_sha256"] == sha(users_path) and binding["user_axis_sha256"] == sha(user_axis_path)
    assert binding["preview_users"] == [] and binding["preview_examples"] == {}
    sequences = load(users_path)
    occurrence_counts = Counter(item for sequence in sequences.values() for item in sequence)
    assert len(sequences) == 24772 and occurrence_counts.total() == 206153 and len(occurrence_counts) == 9922
    assert min(map(len, sequences.values())) == min(occurrence_counts.values()) == 5
    assert all(len(set(sequence)) == len(sequence) for sequence in sequences.values())
    assert sha256(compact(list(sequences.values()))).hexdigest() == binding["sequences_sha256"]
    splits = offline.verify_splits(prepared, sequences)
    assert set(binding["native_readback"]) == {"data/" + name for name in splits}
    for split, result in splits.items():
        observed = binding["native_readback"]["data/" + split]
        assert observed["rows_read_exact"] == result["users"] and observed["sha256"] == result["sha256"]
    assert splits["evaluation"]["stored_items"] - 24772 == 148927
    assert splits["testing"]["stored_items"] - 24772 == 172929
    return proof | {"colliding_prefixes": sum(n > 1 for n in prefixes.values()),
                    "items_in_colliding_prefixes": sid["items_in_colliding_prefixes"],
                    "all_predicted_center_ids_equal_sid_prefix": len(codes) * 3,
                    "all_upstream_record_rows": len(identities), "users": len(sequences), "splits": splits}


def verify_configuration(actual, planned, substitutions, stage):
    import yaml

    def relocate(value):
        if isinstance(value, str):
            for source, destination in substitutions.items():
                value = value.replace(source, destination)
            return value
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        return value

    got = yaml.safe_load(actual.read_bytes())
    want = relocate(yaml.safe_load(planned.read_bytes()))
    got_id, want_id = got.pop("id"), want.pop("id")
    assert got == want, [key for key in got.keys() | want.keys() if got.get(key) != want.get(key)]
    return {"stage": stage, "actual_sha256": sha(actual), "resolved_plan_sha256": sha(planned),
            "runtime_id": got_id, "planned_id": want_id, "path_substitutions": substitutions}


def reassign(run, common, checkpoint, request, offline):
    import numpy as np

    upstream = run / "upstream"
    fit = load(upstream / "fit_gate.json")
    original = load(checkpoint / "checkpoint_extraction.json")
    state = checkpoint / "quantizer_native_state.pt"
    selected = load(checkpoint / "capture.json")["files"][state.name]
    assert original["before"] == original["after"] and original["mode"] == "formal"
    assert original["checkpoint_sha256"] == fit["checkpoint_sha256"]
    assert original["metadata"]["global_step"] == 30 and original["metadata"]["layers_initialized"] == [True] * 3
    assert sha(state) == selected["sha256"] and state.stat().st_size == selected["bytes"]
    book_by_hash = {}
    with ZipFile(state) as archive:
        for name in archive.namelist():
            if "/data/" in name:
                data = archive.read(name)
                book_by_hash[sha256(data).hexdigest()] = data
        (byteorder,) = [name for name in archive.namelist() if name.endswith("/byteorder")]
        assert archive.read(byteorder) == b"little"
    codebooks = []
    for layer in range(3):
        info = original["codebooks"][f"quantization_layer_list.{layer}.centroids"]
        assert info["dtype"] == "torch.float32" and info["shape"] == [256, 3584]
        data = book_by_hash[info["sha256"]]
        centres = np.frombuffer(data, dtype="<f4").reshape(info["shape"]).astype(np.float64)
        assert np.isfinite(centres).all()
        codebooks.append(centres)

    path = common / "item_embeddings.pt"
    assert sha(path) == request["scope"]["input_hashes"][path.name]
    with ZipFile(path) as archive:
        (name,) = [name for name in archive.namelist() if name.endswith("/data/0")]
        assert archive.read(name.split("/")[0] + "/byteorder") == b"little"
        vectors = np.frombuffer(archive.read(name), dtype="<f4").reshape(9922, 3584)
    assert np.isfinite(vectors).all()
    hashes = load_rows(common / "item_identity.jsonl")
    assert all(sha256(vector).hexdigest() == row["embedding_row_sha256"]
               for vector, row in zip(vectors, hashes, strict=True))
    obs = offline.archive_i64(upstream / "sid_generation/predicted_codes_by_item.pt", (9922, 3), (3, 1))
    actual = np.asarray(obs).reshape(9922, 3)
    mismatched = [[], [], []]
    min_margin = [float("inf")] * 3
    centre_norms = [np.sum(book * book, axis=1) for book in codebooks]
    for start in range(0, 9922, 256):
        stop = start + 256
        residual = vectors[start:stop].astype(np.float64)
        for level, (book, squared) in enumerate(zip(codebooks, centre_norms, strict=True)):
            residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
            # The common squared input norm does not affect an argmin.
            distance = squared - 2 * (residual @ book.T)
            nearest = distance.argmin(axis=1)
            want = actual[start:stop, level]
            mismatched[level].extend((start + int(i)) for i in np.flatnonzero(nearest != want))
            first_two = np.partition(distance, 1, axis=1)[:, :2]
            min_margin[level] = min(min_margin[level], float(np.min(first_two[:, 1] - first_two[:, 0])))
            residual -= book[want]
    assert not any(mismatched), mismatched
    return {"embedding_rows": len(vectors), "embedding_dtype": str(vectors.dtype), "centres_per_layer": 256,
            "actual_code_comparisons": actual.size, "mismatched_items_by_layer": mismatched,
            "minimum_first_second_squared_distance_margin_by_layer": min_margin,
            "source_checkpoint_sha256": original["checkpoint_sha256"],
            "extracted_state_sha256": selected["sha256"]}


def verify_training(cutoff):
    metrics = sorted(cutoff.rglob("metrics.csv"))
    (validation_path,) = list(cutoff.rglob("validation.jsonl"))
    records = load_rows(validation_path)
    assert records and [row["step"] for row in records] == list(range(100, 100 * (1 + len(records)), 100))
    assert all(row["batches"] == 3097 and 0 <= row["recall_at_5"] <= 1 for row in records)
    train = [p for p in metrics if "training" in p.parts]
    if not train:
        train = metrics
    reviewed_records = []
    for path in train:
        assert path.read_bytes().endswith(b"\n"), path
        with path.open(newline="") as file:
            reviewed_records.extend(csv.DictReader(file))
    assert reviewed_records and all(None not in row for row in reviewed_records)
    for row in reviewed_records:
        assert all(math.isfinite(float(value)) for value in row.values() if value not in ("", None))
    train_losses = [row for row in reviewed_records if row.get("train/loss_step")]
    val_metrics = [row for row in reviewed_records if row.get("val/recall@5")]
    assert train_losses and len(val_metrics) == len(records)
    for row, csv_record in zip(records, val_metrics, strict=True):
        assert row["recall_at_5"] == float(csv_record["val/recall@5"])
    log = (cutoff / "work.log").read_text()
    assert not re.search(r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError", log)
    assert [match.group(1) for match in re.finditer(r"Starting (\w+) \(formal\)", log)] == [
        "quantizer", "sid", "bound", "training"]
    model = load(cutoff / "output/run/training/model.json")
    assert model["parameters"] == model["trainable_parameters"] == 13128064
    assert model["encoder_layers"] == model["decoder_layers"] == 4
    assert list(model["dtype_counts"]) == ["torch.float32"]
    assert model["sid_slots"] == 1024 and model["embedding_width"] == 128
    assert model["shared"] and model["sid_matches_official"]
    assert model["heads"] == [[256, 128]] * 4 and model["beam"] == 10
    assert model["deterministic"] and model["matmul_precision"] == "medium"
    return {"complete_validation_records": records, "validation_csv_records": val_metrics,
            "last_train_loss_step_csv_record": train_losses[-1], "train_loss_step_records": len(train_losses),
            "largest_observed_validation_recall_at_5": max(row["recall_at_5"] for row in records),
            "full_validation_users_from_verified_bound_split": 24772, "model": model}


def verify_cutoff_capture(cutoff):
    manifest = load(cutoff / "capture.json")
    assert manifest["selection"] == "cutoff"
    begin = datetime.fromisoformat(manifest["started_utc"])
    end = datetime.fromisoformat(manifest["ended_utc"])
    changing = []
    for name, record in manifest["files"].items():
        path = cutoff / name
        assert path.is_file() and not path.is_symlink() and cutoff.resolve() in path.resolve().parents
        assert sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], path
        if "before" not in record:
            assert record["kind"] == "observation" and name == "sample.json"
            continue
        assert record["start_byte"] == 0
        assert record["end_byte_exclusive"] == record["bytes"] == record["before"]["bytes"]
        assert begin <= datetime.fromisoformat(record["captured_at"]) <= end
        if record["before"] != record["after"]:
            changing.append(name)
    return {"started_utc": manifest["started_utc"], "ended_utc": manifest["ended_utc"],
            "bounded_entries_verified": len(manifest["files"]), "entries_changed_during_read": changing}


def verify_observation(root, cutoff, request, launch):
    window_start = datetime.fromisoformat(launch["independent_connection_confirmed_at"])
    points = []
    for point, path in (("T10", root / "observations/T10"), ("T20", root / "observations/T20"),
                        ("T30", cutoff)):
        query = load(path / "query.json")
        sample = load(path / "sample.json")
        assert query["request_sha256"] == launch["canonical_request_sha256"]
        assert query["started_at"] == sample["status"]["started_at"] == launch["started_at"]
        assert query["job_id"] == sample["status"]["job_id"] == request["job_id"]
        assert query["status"] == sample["status"]["status"] == "RUNNING"
        assert query["supervisor_alive"] and query["worker_alive"] and query["workgroup_live"]
        for role in ("supervisor", "worker"):
            original = query[role]
            process = sample["processes"][str(original["pid"])]
            assert original == sample["status"][role]
            assert process["start_ticks_match"] and process["tty_nr"] == 0
            assert process["start_ticks"] == original["start_ticks"]
            assert process["pid"] == process["pgid"] == process["sid"] == launch[role + "_pid"]
        gpu = launch["gpu_uuid"]
        gpu_rows = [row for row in csv.reader(sample["gpu"].splitlines()) if row[1].strip() == gpu]
        gpu_apps = [int(row[1]) for row in csv.reader(sample["gpu_apps"].splitlines()) if row[0] == gpu]
        members = {member["pid"] for member in query["workgroup_observation"]["owned_live"]}
        assert gpu_apps and set(gpu_apps) <= members
        (row,) = gpu_rows
        assert "4090" in row[2]
        assert all(value > 2 * 1024**3 for value in sample["free_bytes"].values())
        log = sample["files"]["work.log"]
        assert not re.search(r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError", log["tail"])
        points.append({"point": point, "sample_started_utc": sample["sample_started_utc"],
                       "sample_ended_utc": sample["sample_ended_utc"],
                       "seconds_since_start_confirm": (datetime.fromisoformat(sample["sample_started_utc"]) - window_start).total_seconds(),
                       "work_log_end_byte_exclusive": log["end_byte_exclusive"],
                       "gpu_pids_in_workgroup": gpu_apps, "gpu_used_memory": row[4].strip(),
                       "gpu_utilization": row[5].strip(), "free_bytes": sample["free_bytes"]})
    times = [point["seconds_since_start_confirm"] for point in points]
    assert 0 < times[0] < times[1] < times[2] and times[-1] >= 1800
    return points


def verify_best(cutoff, directory, training):
    state = load(directory / "checkpoint_extraction.json")
    files = load(cutoff / "sample.json")["checkpoints"]
    path = state["source"]
    assert state["before"] == state["after"]
    name = Path(path).name
    assert list(files) == [name]
    assert all(state["before"][field] == files[name][field] for field in state["before"])
    (callback,) = state["callbacks"]
    assert callback["monitor"] == "val/recall@5"
    assert callback["best_model_path"] == path
    assert callback["best_model_score"] == training["largest_observed_validation_recall_at_5"]
    assert callback["best_k_models"] == {path: callback["best_model_score"]}
    assert any(row["step"] == state["global_step"] and row["recall_at_5"] == callback["best_model_score"]
               for row in training["complete_validation_records"])
    (optimizer,) = state["optimizer"]
    assert optimizer["state_entries"] > 0 and optimizer["step_values"] == [state["global_step"]]
    assert all(group["lr"] == 0.001 and group["weight_decay"] == 0.0001 for group in optimizer["param_groups"])
    return {"path": path, "sha256": state["checkpoint_sha256"], "global_step": state["global_step"],
            "epoch": state["epoch"], "monitor": callback["monitor"], "score": callback["best_model_score"],
            "adam_parameter_state_entries": optimizer["state_entries"], "adam_step_values": optimizer["step_values"]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--offline-verifier", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--common", type=Path)
    parser.add_argument("--quantizer-checkpoint", type=Path)
    parser.add_argument("--runtime-plan", type=Path)
    parser.add_argument("--cutoff", type=Path)
    parser.add_argument("--best-checkpoint", type=Path)
    args = parser.parse_args()
    request, launch = verify_launch(args.launch, args.source, args.runner)
    output = {"launch": launch}
    if args.run:
        assert args.common and args.offline_verifier
        spec = importlib.util.spec_from_file_location("independent_wire_format", args.offline_verifier)
        offline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline)
        output["receipts"] = verify_receipts(args.run, request)
        output["fit"] = verify_fit(args.run / "upstream", request)
        output["catalogue"] = verify_catalogue(args.run, args.common, request, offline)
        if args.quantizer_checkpoint:
            output["independent_reassignment"] = reassign(args.run, args.common, args.quantizer_checkpoint,
                                                            request, offline)
    if args.cutoff:
        output["cutoff_capture"] = verify_cutoff_capture(args.cutoff)
        output["training"] = verify_training(args.cutoff)
        output["observations"] = verify_observation(args.launch, args.cutoff, request, launch)
        if args.best_checkpoint:
            output["best_at_cutoff"] = verify_best(args.cutoff, args.best_checkpoint, output["training"])
    if args.runtime_plan:
        assert args.run is not None
        planned_root = str(Path(request["cwd"]).parent / "formal-runtime-plan")
        root = load(args.launch / "start-query.json")["result_path"] + "/run"
        changes = {planned_root + "/upstream": root + "/upstream",
                   planned_root + "/bound": root + "/bound",
                   planned_root + "/downstream": root + "/training"}
        for stage in ("quantizer_train", "sid_generation"):
            relative = Path("upstream", stage, "resolved.yaml")
            output.setdefault("resolved_runtime_config", []).append(verify_configuration(
                args.run / relative, args.runtime_plan / relative, changes, stage))
        if args.cutoff:
            output["resolved_runtime_config"].append(verify_configuration(
                args.cutoff / "output/run/training/effective_config.yaml",
                args.runtime_plan / "downstream/effective_config.yaml", changes, "training"))
    output["reviewed_at"] = datetime.now().astimezone().isoformat()
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
