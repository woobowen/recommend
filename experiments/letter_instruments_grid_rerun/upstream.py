"""Prepare and verify the GRID RQ-KMeans product catalogue."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent / "upstream"
ITEMS = 9922
DIMENSIONS = 3584
WIDTH = 256
LAYERS = 3
FIT_STEPS = {"formal": 30, "preview": 6}
INPUT_MANIFEST = "upstream_inputs.json"
FIT_GATE = "fit_gate.json"
SID_GATE = "sid_gate.json"
SID_TENSOR = "semantic_ids.pt"
SID_JSONL = "item_sids.jsonl"


def digest(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def digest_stream(stream):
    sha = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    scratch = path.with_name(path.name + ".partial")
    scratch.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scratch.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, rows):
    scratch = path.with_name(path.name + ".partial")
    with scratch.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    scratch.replace(path)


def input_identity(embedding, identity):
    return {
        "item_embeddings_sha256": digest(embedding),
        "item_identity_sha256": digest(identity),
        "items_qwen_sha256": digest(identity.with_name("items_qwen.jsonl")),
    }


def identity_rows(identity):
    rows = read_rows(identity)
    if len(rows) != ITEMS:
        raise ValueError(f"Expected {ITEMS} item identities, found {len(rows)}")
    for expected_id, row in enumerate(rows):
        if type(row["item_id"]) is not int or row["item_id"] != expected_id:
            raise ValueError(f"Item identity row {expected_id} must contain internal ID {expected_id}")
        if type(row["letter_item_id"]) is not int or row["letter_item_id"] != expected_id:
            raise ValueError(f"LETTER ID differs from internal ID {expected_id}")
        if type(row["old_grid_embedding_row"]) is not int or not 0 <= row["old_grid_embedding_row"] < 10611:
            raise ValueError(f"Invalid old GRID embedding row for item {expected_id}")
        if type(row["old_grid_item_id"]) is not int or row["old_grid_item_id"] != row["old_grid_embedding_row"]:
            raise ValueError(f"Old GRID item ID differs from the embedding row for item {expected_id}")
        if not isinstance(row["asin"], str) or not row["asin"]:
            raise ValueError(f"Missing ASIN for item {expected_id}")
    for key in ("letter_item_id", "asin", "old_grid_item_id", "old_grid_embedding_row"):
        if len({row[key] for row in rows}) != ITEMS:
            raise ValueError(f"Duplicate product identity in column {key}")
    return rows


def validate_embeddings(embedding, rows):
    tensor = torch.load(embedding, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (ITEMS, DIMENSIONS):
        raise ValueError(f"Expected a [{ITEMS},{DIMENSIONS}] frozen embedding tensor")
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("Frozen embeddings must be finite float32 values")
    for position, row in enumerate(rows):
        raw = tensor[position].contiguous().numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != row["embedding_row_sha256"]:
            raise ValueError(f"Frozen embedding row mismatch for internal ID {position}, ASIN {row['asin']}")
    return {
        "rows_verified": len(rows),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "all_finite": True,
        "tensor_order": "ascending internal item_id",
    }


def write_item_records(output, identity, rows):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf

    # The source writer kept TensorFlow on CPU. The child GRID process uses the selected CUDA device.
    tf.config.set_visible_devices([], "GPU")
    texts = read_rows(identity.with_name("items_qwen.jsonl"))
    if len(texts) != ITEMS:
        raise ValueError("The selected Qwen text file does not cover the catalogue")
    record_path = output / "catalogue" / "items" / "part-00000.tfrecord.gz"
    record_path.parent.mkdir(parents=True)
    scratch = record_path.with_name(record_path.name + ".partial")
    with tf.io.TFRecordWriter(str(scratch), options="GZIP") as writer:
        for item_id, (row, item) in enumerate(zip(rows, texts)):
            if item["item_id"] != item_id:
                raise ValueError(f"Qwen text item ID mismatch at row {item_id}")
            value = item["text"].encode("utf-8")
            if hashlib.sha256(value).hexdigest() != row["text_sha256"]:
                raise ValueError(f"Frozen encoding text mismatch for item {item_id}")
            features = {
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id])),
                "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
    dataset = tf.data.TFRecordDataset(str(scratch), compression_type="GZIP")
    count = 0
    for position, content in enumerate(dataset.as_numpy_iterator()):
        if position >= ITEMS:
            raise ValueError("Excess upstream TFRecord rows")
        features = tf.train.Example.FromString(content).features.feature
        if list(features["id"].int64_list.value) != [position]:
            raise ValueError(f"Upstream TFRecord changed an item ID at row {position}")
        if list(features["text"].bytes_list.value) != [texts[position]["text"].encode("utf-8")]:
            raise ValueError(f"Upstream TFRecord changed encoding text at row {position}")
        count += 1
    if count != ITEMS:
        raise ValueError(f"Upstream TFRecord readback produced {count} items")
    scratch.replace(record_path)
    return record_path


def override_path(key, path):
    return f'{key}={json.dumps(str(path), ensure_ascii=False)}'


def native_command(output, embedding, mode, stage, checkpoint=None):
    out = output / stage
    inference = stage == "sid_generation"
    prefix = "predict" if inference else "train"
    args = [
        sys.executable, str(ROOT / "runtime.py"),
        "experiment=rkmeans_inference_flat" if inference else "experiment=rkmeans_train_flat",
        "++callbacks.restart_job=null",
        override_path("data_dir", output / "catalogue"),
        override_path("paths.root_dir", ROOT / "native"),
        "trainer.devices=1", "++trainer.strategy=auto", "trainer.precision=32-true",
        "extras.print_config=false", override_path("embedding_path", embedding),
        f"embedding_dim={DIMENSIONS}", f"num_hierarchies={LAYERS}", f"codebook_width={WIDTH}",
        "seed=42", f"++data_loading.datamodule.{prefix}_dataloader_config.num_workers=0",
        f"++data_loading.datamodule.{prefix}_dataloader_config.timeout=0",
        f"++data_loading.datamodule.{prefix}_dataloader_config.persistent_workers=false",
        override_path("hydra.run.dir", out), "++trainer.enable_progress_bar=false",
    ]
    if inference:
        args.extend([override_path("ckpt_path", checkpoint), "callbacks.bq_writer=null"])
    else:
        args.extend(["train=true", "test=false", "ckpt_path=null"])
        if mode == "preview":
            args.append(f"trainer.max_steps={FIT_STEPS[mode]}")
    return args


def native_environment():
    env = dict(os.environ)
    for key in ("PROJECT_ROOT", "METADATA_DIR", "LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GRID_UPSTREAM_CONFIG_ONLY"):
        env.pop(key, None)
    env.update(
        PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1", TF_CPP_MIN_LOG_LEVEL="3",
        TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
    )
    return env


def run_native(output, embedding, mode, stage, checkpoint=None, config_only=False):
    args = native_command(output, embedding, mode, stage, checkpoint)
    cwd = ROOT / "native"
    log_path = output / f"{stage}.log"
    env = native_environment()
    if config_only:
        env.update(GRID_UPSTREAM_CONFIG_ONLY="1", CUDA_VISIBLE_DEVICES="")
    atomic_json(output / f"{stage}.command.json", {
        "argv": args, "cwd": str(cwd), "mode": mode, "visible_gpus": env.get("CUDA_VISIBLE_DEVICES"),
        "config_only": config_only,
    })
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    atomic_json(output / f"{stage}.exit.json", {
        "exit_code": process.returncode, "elapsed_s": time.monotonic() - started,
    })
    process.check_returncode()


def validate_fit(output, mode, rows):
    folder = output / "quantizer_train"
    result = read_json(folder / "stage_result.json")
    steps = FIT_STEPS[mode]
    checkpoint = Path(result["checkpoint"]).resolve()
    if checkpoint.parent != (folder / "checkpoints").resolve() or not checkpoint.is_file():
        raise ValueError("The quantizer checkpoint is not from the current fit stage")
    if result["global_step"] != steps:
        raise ValueError(f"Expected exactly {steps} trainer steps")
    saved = read_json(folder / "checkpoint_evidence.json")
    if saved["global_step"] != steps or saved["layers_initialized"] != [True] * LAYERS:
        raise ValueError("The fit checkpoint does not contain every initialized quantization layer")
    if saved["current_layer"] != LAYERS - 1:
        raise ValueError("The fit checkpoint did not reach the final residual layer")
    if saved["centroid_shapes"] != [[WIDTH, DIMENSIONS]] * LAYERS or not all(saved["centroids_finite"]):
        raise ValueError("Malformed or non-finite native codebook tensors")

    events = read_rows(folder / "fit_batches.jsonl")
    per_layer = {layer: {"initialization_batches": 0, "initializations": 0, "updates": 0, "changed_updates": 0}
                 for layer in range(LAYERS)}
    all_items = set()
    for index, event in enumerate(events):
        if event["batch_idx"] != index or event["batch_size"] != 2048:
            raise ValueError(f"Unexpected native batch index or size at position {index}")
        if len(event["item_ids"]) != event["batch_size"] or not all(0 <= item_id < len(rows) for item_id in event["item_ids"]):
            raise ValueError(f"Invalid native fit item axis at batch {index}")
        if not math.isfinite(event["loss"]) or not all(event["centroids_finite"]):
            raise ValueError(f"Native fit batch {index} contains non-finite values")
        all_items.update(event["item_ids"])
        layer = event["layer"]
        counters = per_layer[layer]
        changed = event["centroids_changed"]
        if any(changed[other] for other in range(LAYERS) if other != layer):
            raise ValueError(f"An inactive codebook changed at native batch {index}")
        delta = event["step_after"] - event["step_before"]
        if delta == 0:
            counters["initialization_batches"] += 1
            if event["initializing"]:
                counters["initializations"] += 1
                if not changed[layer] or event["initialization_buffer_before"] != 2048 or event["initialization_buffer_after"] != 0:
                    raise ValueError(f"Missing or malformed initializer update for layer {layer}")
            elif changed[layer] or event["initialization_buffer_before"] != 0 or event["initialization_buffer_after"] != 2048:
                raise ValueError(f"Unexpected centroid or buffer activity before initialization of layer {layer}")
        elif delta == 1 and event["initialized_after"] and not event["initializing"]:
            counters["updates"] += 1
            counters["changed_updates"] += int(changed[layer])
        else:
            raise ValueError(f"Unexpected optimizer transition at native batch {index}")
    if all_items != set(range(len(rows))):
        raise ValueError("Bounded fit did not observe the entire common catalogue")
    for layer, counts in per_layer.items():
        if counts["initialization_batches"] != 2 or counts["initializations"] != 1:
            raise ValueError(f"Initialization coverage failed for layer {layer}: {counts}")
        if counts["updates"] != steps // LAYERS or not counts["changed_updates"]:
            raise ValueError(f"Tracked update coverage failed for layer {layer}: {counts}")
    if len(events) != steps + LAYERS * 2:
        raise ValueError(f"Unexpected aggregate raw fit batch count {len(events)}")
    return checkpoint, {
        "layers_initialized": saved["layers_initialized"], "step": steps,
        "raw_batches": len(events), "rows_processed_with_repeats": sum(e["batch_size"] for e in events),
        "distinct_rows_observed": len(all_items), "per_layer": per_layer,
        "observed_elapsed_s": events[-1]["fit_elapsed_s"],
    }


def fit(output, mode, embedding, identity):
    output.mkdir(parents=True, exist_ok=False)
    identities = identity_rows(identity)
    identity_hashes = input_identity(embedding, identity)
    emb_evidence = validate_embeddings(embedding, identities)
    item_records = write_item_records(output, identity, identities)
    atomic_json(output / INPUT_MANIFEST, {
        **identity_hashes, "catalogue_tfrecord_sha256": digest(item_records),
        "embedding": emb_evidence, "item_records_readback": len(identities), "mode": mode,
    })
    run_native(output, embedding, mode, "quantizer_train")
    checkpoint, stats = validate_fit(output, mode, identities)
    atomic_json(output / FIT_GATE, {
        "mode": mode, "input_identity": identity_hashes, "checkpoint_sha256": digest(checkpoint),
        "checkpoint": str(checkpoint.relative_to(output)), "native_checkpoint": str(checkpoint), **stats,
    })


def verify_fit_binding(output, mode, embedding, identity):
    gate = read_json(output / FIT_GATE)
    inputs = read_json(output / INPUT_MANIFEST)
    if gate["mode"] != mode or inputs["mode"] != mode or gate["step"] != FIT_STEPS[mode]:
        raise ValueError(f"The fit gate does not authorize a {mode} downstream stage")
    if gate["layers_initialized"] != [True] * LAYERS:
        raise ValueError("Ineligible quantizer checkpoint because initialization is incomplete")
    current = input_identity(embedding, identity)
    if gate["input_identity"] != current or any(inputs[key] != value for key, value in current.items()):
        raise ValueError("The frozen input identity differs from the executed fit")
    result = read_json(output / "quantizer_train" / "stage_result.json")
    checkpoint = (output / gate["checkpoint"]).resolve()
    if checkpoint.parent != (output / "quantizer_train" / "checkpoints").resolve():
        raise ValueError("Quantizer checkpoint path escapes this fit stage")
    original = Path(result["checkpoint"])
    relative = Path(gate["checkpoint"])
    if not original.is_absolute() or str(original) != gate["native_checkpoint"]:
        raise ValueError("Original quantizer checkpoint path differs from the fit evidence")
    if tuple(original.parts[-len(relative.parts):]) != relative.parts or digest(checkpoint) != gate["checkpoint_sha256"]:
        raise ValueError("Current quantizer checkpoint path suffix or hash differs from the fit gate")
    record = output / "catalogue" / "items" / "part-00000.tfrecord.gz"
    if digest(record) != inputs["catalogue_tfrecord_sha256"]:
        raise ValueError("Upstream TFRecord differs from the data used in the fitted quantizer")
    return checkpoint, current


def validate_codes(sidecar, codes, expected_items=ITEMS):
    """Check the native deduplication against a per-prefix item-index oracle."""
    if codes.dtype != torch.int64 or codes.shape != (LAYERS, expected_items):
        raise ValueError(f"Expected [{LAYERS},{expected_items}] int64 codes prior to deduplication")
    if sidecar.dtype != torch.int64 or sidecar.shape != (LAYERS + 1, expected_items):
        raise ValueError(f"Expected [{LAYERS + 1},{expected_items}] int64 SIDs")
    if torch.any(codes < 0) or torch.any(codes >= WIDTH):
        raise ValueError("Residual code index is outside its existing level capacity")
    if not torch.equal(sidecar[:LAYERS], codes):
        raise ValueError("SID output changed item axis or predictions in the first three levels")
    if torch.any(sidecar[LAYERS] < 0) or torch.any(sidecar[LAYERS] >= WIDTH):
        raise ValueError("Prefix collision count exceeds the existing fourth-level capacity")
    groups = defaultdict(list)
    for item, prefix in enumerate(codes.T.tolist()):
        groups[tuple(prefix)].append(item)
    expected_fourth = [0] * expected_items
    for group in groups.values():
        if len(group) > 1:
            for ordinal, item in enumerate(group, start=1):
                expected_fourth[item] = ordinal
    if sidecar[LAYERS].tolist() != expected_fourth:
        raise ValueError("Fourth SID code differs from the original item-axis deduplication")
    if len({tuple(sid) for sid in sidecar.T.tolist()}) != expected_items:
        raise ValueError("Final SIDs do not uniquely identify catalogue items")
    groups_histogram = Counter(len(group) for group in groups.values())
    return {
        "items": expected_items, "sid_shape": list(sidecar.shape),
        "unique_sids": expected_items, "unique_prefixes": len(groups),
        "prefix_group_size_histogram": {str(size): count for size, count in sorted(groups_histogram.items())},
        "items_in_colliding_prefixes": sum(size * count for size, count in groups_histogram.items() if size > 1),
        "digit_ranges": [
            {"min": int(values.min()), "max": int(values.max()), "used": int(values.unique().numel())}
            for values in sidecar
        ],
    }


def make_item_sids(rows, sids):
    return [
        {"item_id": row["item_id"], "letter_item_id": row["letter_item_id"], "asin": row["asin"], "sid": sid}
        for row, sid in zip(rows, sids.T.tolist())
    ]


def load_prediction_codes(output):
    return torch.load(
        output / "sid_generation" / "predicted_codes_by_item.pt", map_location="cpu", weights_only=True,
    ).T


def generate_sids(output, mode, embedding, identity):
    checkpoint, input_hashes = verify_fit_binding(output, mode, embedding, identity)
    rows = identity_rows(identity)
    run_native(output, embedding, mode, "sid_generation", checkpoint)
    values = torch.load(
        output / "sid_generation" / "pickle" / "merged_predictions_tensor.pt",
        map_location="cpu", weights_only=True,
    )
    codes = load_prediction_codes(output)
    stats = validate_codes(values, codes)
    stats["prediction"] = read_json(output / "sid_generation" / "prediction_observation.json")
    torch.save(values, output / SID_TENSOR)
    write_rows(output / SID_JSONL, make_item_sids(rows, values))
    atomic_json(output / SID_GATE, {
        **stats,
        "mode": mode, "input_identity": input_hashes, "fit_gate_sha256": digest(output / FIT_GATE),
        "checkpoint_sha256": digest(checkpoint), "sid_sha256": digest(output / SID_TENSOR),
        "item_sids_sha256": digest(output / SID_JSONL),
        "predicted_codes_sha256": digest(output / "sid_generation" / "predicted_codes_by_item.pt"),
    })
    verify_sid_bundle(output, mode, embedding, identity)


def verify_sid_bundle(output, mode, embedding, identity):
    """Return the current catalogue gate after verifying provenance and the exact SID mapping."""
    output, embedding, identity = (Path(path).resolve(strict=True) for path in (output, embedding, identity))
    checkpoint, input_hashes = verify_fit_binding(output, mode, embedding, identity)
    gate = read_json(output / SID_GATE)
    expected_files = {
        "fit_gate_sha256": output / FIT_GATE, "checkpoint_sha256": checkpoint,
        "sid_sha256": output / SID_TENSOR, "item_sids_sha256": output / SID_JSONL,
        "predicted_codes_sha256": output / "sid_generation" / "predicted_codes_by_item.pt",
    }
    if gate["mode"] != mode or gate["input_identity"] != input_hashes:
        raise ValueError(f"SID input identity or scope does not match the requested {mode} run")
    for key, path in expected_files.items():
        if gate[key] != digest(path):
            raise ValueError(f"SID bundle identity mismatch for {path.name}")
    sids = torch.load(output / SID_TENSOR, map_location="cpu", weights_only=True)
    stats = validate_codes(sids, load_prediction_codes(output))
    if stats != {key: gate[key] for key in stats}:
        raise ValueError("SID gate no longer agrees with the current SID counts and ranges")
    if read_rows(output / SID_JSONL) != make_item_sids(identity_rows(identity), sids):
        raise ValueError("SID item lookup changed or is not on the full current item axis")
    return gate


def resolve_configs(output, mode, embedding, identity):
    """Render the real Hydra dispatch without constructing a trainer or reading assets."""
    output.mkdir(parents=True, exist_ok=False)
    run_native(output, embedding, mode, "quantizer_train", config_only=True)
    example_checkpoint = output / "quantizer_train" / "checkpoints" / f"checkpoint_000_{FIT_STEPS[mode]:06d}.ckpt"
    run_native(output, embedding, mode, "sid_generation", example_checkpoint, config_only=True)
    return {"mode": mode, "checkpoint_resolution": "fit gate supplies the actual checkpoint at runtime"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "sid", "verify", "config"))
    parser.add_argument("--mode", required=True, choices=tuple(FIT_STEPS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    args = parser.parse_args()
    action = {"fit": fit, "sid": generate_sids, "verify": verify_sid_bundle, "config": resolve_configs}[args.stage]
    requires_assets = args.stage != "config"
    result = action(
        args.output.resolve(), args.mode, args.embedding.resolve(strict=requires_assets),
        args.identity.resolve(strict=requires_assets),
    )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
