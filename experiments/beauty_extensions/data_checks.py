"""Fix an attention cohort and audit complete Beauty test metadata on CPU.

Both functions follow the selected SequenceDataModule file assignment. They
never instantiate a model, alter TFRecords/SIDs, or inspect attention or hits.
Published records contain the same opaque user keys as PredictionRecorder.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class _Row:
    user: int
    items: tuple[int, ...]
    file_index: int
    row_index: int
    global_row_index: int


def _key(user: int) -> str:
    return hashlib.sha256(("beauty-extensions-v1:" + str(int(user))).encode()).hexdigest()[:24]


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if json.loads(path.read_text()) != value:
        raise ValueError("Persisted data-check JSON differs from the written record")


def _local_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
        raise ValueError("This Beauty check requires local server TFRecord files")
    return Path(unquote(parsed.path) if parsed.scheme else uri)


def _loader_and_sid(cfg, spec, split):
    import hydra
    import torch
    from .common import verify_asset
    from src.data.loading.utils import assign_files_to_workers
    from src.utils.file_utils import list_files

    verify_asset(spec["sid"])
    sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True)
    if not isinstance(sid, torch.Tensor) or sid.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ) or sid.ndim != 2 or sid.shape[0] != 4 or sid.numel() == 0 or bool((sid < 0).any()):
        raise ValueError("Expected the verified nonnegative integer [4,items] SID tensor")
    lc = hydra.utils.instantiate(cfg.data_loading[f"{split}_dataloader_config"].dataloader)
    if lc.should_shuffle_rows or getattr(lc, "limit_files", None):
        raise ValueError("Full evaluation metadata must not shuffle or limit files")
    if lc.drop_last or not lc.dataset_config.iterate_per_row:
        raise ValueError("Expected complete row-wise evaluation with drop_last=False")
    if lc.padding_token != -1 or lc.masking_token != -1 or lc.get("oov_token", None) is not None:
        raise ValueError("The confirmed Beauty data protocol requires padding/masking=-1 and no OOV removal")
    if lc.sequence_length % 4 or lc.sequence_length < 8:
        raise ValueError("Sequence length must hold at least two complete H4 item groups")
    if set(lc.labels) != {"sequence_data"}:
        raise ValueError("Expected the original single sequence_data target")
    label = lc.labels["sequence_data"].transform
    if type(label).__name__ != "NextKTokenMasking" or label.next_k != 4:
        raise ValueError("Expected the original NextKTokenMasking(next_k=4) target contract")
    dataset_sid = lc.dataset_config.semantic_id_map["sequence_data"]
    if not torch.equal(dataset_sid, sid):
        raise ValueError("Hydra dataset SID mapping differs from the verified selected SID")
    suffix = getattr(lc.dataset_config, "file_format", None) or lc.dataset_config.data_iterator.get_file_suffix()
    enumerated = list_files(folder_path=lc.data_folder, suffix="*" + suffix)
    assigned, shared = assign_files_to_workers(
        list_of_files=enumerated, total_workers=1,
        assign_by_size=lc.assign_files_by_size,
        should_shuffle_rows=lc.should_shuffle_rows,
        assign_all_files_per_worker=getattr(lc, "assign_all_files_per_worker", False),
    )
    files = assigned[0]
    if not files or len(files) != len(set(files)):
        raise ValueError("Evaluation files are missing or duplicated in the assignment")
    identities = []
    for uri in files:
        path = _local_path(uri)
        stat = path.stat()
        if not path.is_file():
            raise ValueError("Assigned TFRecord is not a readable regular file")
        identities.append({"path": uri, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "device": stat.st_dev, "inode": stat.st_ino})
    metadata = {
        "total_workers": 1, "assign_files_by_size": lc.assign_files_by_size,
        "assign_all_files_per_worker": getattr(lc, "assign_all_files_per_worker", False),
        "shared_files_returned": shared, "num_workers": lc.num_workers,
        "batch_size_per_device": lc.batch_size_per_device, "drop_last": lc.drop_last,
        "sequence_length": lc.sequence_length, "file_order": files,
        "enumerated_file_order": enumerated, "files": identities,
        "file_assignment": "SequenceDataModule.setup -> assign_files_to_workers(total_workers=1)",
    }
    return lc, sid, metadata


def _sources_unchanged(metadata):
    for identity in metadata["files"]:
        stat = _local_path(identity["path"]).stat()
        actual = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
        expected = tuple(identity[key] for key in ("bytes", "mtime_ns", "device", "inode"))
        if actual != expected:
            raise RuntimeError("An assigned TFRecord changed during this data check")


def _iter_metadata(files):
    import tensorflow as tf

    global_index = 0
    for file_index, uri in enumerate(files):
        path = _local_path(uri)
        before = path.stat()
        records = tf.data.TFRecordDataset([uri], compression_type="GZIP", num_parallel_reads=1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        records = records.with_options(options)
        for row_index, serialized in enumerate(records):
            example = tf.train.Example()
            example.ParseFromString(serialized.numpy())
            features = example.features.feature
            for name in ("user_id", "sequence_data"):
                if name not in features or features[name].WhichOneof("kind") != "int64_list":
                    raise ValueError(f"Invalid {name} at file {file_index}, row {row_index}")
            users = tuple(features["user_id"].int64_list.value)
            items = tuple(features["sequence_data"].int64_list.value)
            if len(users) != 1 or users[0] < 0 or len(items) < 2:
                raise ValueError(f"Invalid user/next-item record at file {file_index}, row {row_index}")
            yield _Row(int(users[0]), items, file_index, row_index, global_index)
            global_index += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError("TFRecord changed while its metadata was being read")


def _validate_items(row, item_count):
    if min(row.items) < 0 or max(row.items) >= item_count:
        raise ValueError(f"Item index outside selected SID axis at file {row.file_index}, row {row.row_index}")


def _expected(row, sid, length):
    """Independent item-axis construction, including the hidden target slot."""
    import torch

    _validate_items(row, sid.shape[1])
    window = row.items[-(length // 4):]
    padded = torch.full((length,), -1, dtype=sid.dtype)
    padded[:len(window) * 4] = sid[:, list(window)].t().reshape(-1)
    target = sid[:, row.items[-1]].clone()
    hidden = padded.clone()
    hidden[(len(window) - 1) * 4:len(window) * 4] = -1
    mask = (hidden != -1).long()
    history_items = list(window[:-1]) + [-1] * (length // 4 - len(window) + 1)
    return padded, hidden, mask, target, history_items


def _source_targets(rows, lc, sid):
    """Execute the original label object on CPU metadata-derived SID windows."""
    import torch

    expected = [_expected(row, sid, lc.sequence_length) for row in rows]
    original = torch.stack([entry[0] for entry in expected])
    transformed = lc.labels["sequence_data"].transform.transform_label(
        sequence=original, padding_token=lc.padding_token, masking_token=lc.masking_token,
    )
    targets = transformed.labels.reshape(len(rows), 4)
    if not torch.equal(targets, torch.stack([entry[3] for entry in expected])):
        raise ValueError("Original label object does not target the raw record's last item SID")
    if not torch.equal(transformed.sequence, torch.stack([entry[1] for entry in expected])):
        raise ValueError("Original label object did not hide the complete last target item")
    return targets, expected


def prepare_fixed_attention_samples(cfg, spec, output) -> Path:
    """Persist evaluation's first 32 distinct users before any attention/hit view."""
    import tensorflow as tf
    import torch

    output = Path(output)
    if cfg.model.get("num_user_bins", None) is not None:
        raise ValueError("This cohort export omits raw user input and requires num_user_bins=null")
    lc, sid, metadata = _loader_and_sid(cfg, spec, "val")
    if set(lc.dataset_config.features_to_consider) - {"sequence_data", "user_id"}:
        raise ValueError("The selected preprocessing requires additional fields beyond the confirmed SID/user protocol")
    chosen, seen, scanned = [], set(), 0
    for row in _iter_metadata(metadata["file_order"]):
        scanned += 1
        _validate_items(row, sid.shape[1])
        if row.user in seen:
            continue
        seen.add(row.user)
        chosen.append(row)
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise ValueError(f"Only {len(chosen)} different evaluation users exist; required fixed cohort is 32")
    collate = partial(lc.collate_fn, labels=lc.labels, sequence_length=lc.sequence_length,
                      masking_token=lc.masking_token, padding_token=lc.padding_token,
                      oov_token=lc.get("oov_token", None))
    samples, all_inputs, all_masks, all_targets = [], [], [], []
    batch_size = int(lc.batch_size_per_device)
    if batch_size < 1:
        raise ValueError("Invalid evaluation batch size")
    for start in range(0, len(chosen), batch_size):
        rows = chosen[start:start + batch_size]
        processed = []
        for row in rows:
            raw = {}
            for name, values in (("sequence_data", row.items), ("user_id", (row.user,))):
                raw[name] = tf.SparseTensor(indices=[[index] for index in range(len(values))],
                    values=tf.constant(values, dtype=tf.int64), dense_shape=[len(values)])
            for pre in lc.dataset_config.preprocessing_functions:
                raw = pre(raw, dataset_config=lc.dataset_config)
                if raw is None:
                    raise ValueError("Selected evaluation user was filtered by actual preprocessing")
            if not torch.equal(raw["sequence_data"], sid[:, list(row.items)].t().reshape(-1)):
                raise ValueError("Actual preprocessing differs from the selected item-to-SID mapping")
            processed.append(raw)
        inputs, labels = collate(processed)
        x = inputs.transformed_sequences["sequence_data"]
        y = labels.labels["sequence_data"].reshape(len(rows), 4)
        source_targets, expected = _source_targets(rows, lc, sid)
        if not torch.equal(y, source_targets) or not torch.equal(x, torch.stack([entry[1] for entry in expected])):
            raise ValueError("Actual collate input/target differs from raw item-axis construction")
        if not torch.equal(inputs.mask, torch.stack([entry[2] for entry in expected])):
            raise ValueError("Actual collate mask differs from complete visible/hidden item groups")
        if x.dtype not in (torch.int32, torch.int64) or y.dtype not in (torch.int32, torch.int64):
            raise ValueError("Actual collated SID input/target does not have integer dtype")
        all_inputs.extend(x.tolist()); all_masks.extend(inputs.mask.tolist()); all_targets.extend(y.tolist())
        for index, row in enumerate(rows):
            samples.append({"sample_key": _key(row.user), "source_file_index": row.file_index,
                "row_index": row.row_index, "global_row_index": row.global_row_index,
                "raw_item_count": len(row.items), "visible_history_item_ids": expected[index][4],
                "target": y[index].tolist(), "target_hidden": True,
                "input_dtype": str(x.dtype), "target_dtype": str(y.dtype)})
    _sources_unchanged(metadata)
    document = {"schema": "beauty-fixed-attention-samples-v1", "split": "evaluation",
        "selection_rule": "First 32 distinct user_id values in the actual single-GPU file assignment and TFRecord row order; chosen before attention/hits",
        "selected_users": 32, "metadata_rows_scanned": scanned, "model_outputs_inspected": False,
        "sid": dict(spec["sid"]), "sid_dtype": str(sid.dtype), "loader": metadata,
        "sample_keys": [entry["sample_key"] for entry in samples], "input_ids": all_inputs,
        "attention_mask": all_masks, "targets": all_targets, "user_id": None,
        "samples": samples, "raw_user_ids_exported": False}
    path = output / "fixed_attention_samples.json"
    _write_new(path, document)
    return path


def scan_test_coverage(cfg, spec, output) -> dict:
    """Read every testing record once, execute original labels, retain all rows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lc, sid, metadata = _loader_and_sid(cfg, spec, "test")
    index_path = output / "test_target_index.jsonl"
    users, pairs, files = Counter(), Counter(), Counter()
    stream_digest = hashlib.sha256()
    rows_count, item_count, minimum, maximum = 0, 0, None, None
    shortest_history, longest_history, first_truncated = None, None, None
    pending = []
    with index_path.open("x") as stream:
        def record_batch(rows):
            nonlocal rows_count, item_count, minimum, maximum
            nonlocal shortest_history, longest_history, first_truncated
            targets, expected = _source_targets(rows, lc, sid)
            for row, target, tensors in zip(rows, targets.tolist(), expected):
                key = _key(row.user)
                value = {"sample_key": key, "target": target, "source_file_index": row.file_index,
                    "row_index": row.row_index, "global_row_index": row.global_row_index,
                    "raw_item_count": len(row.items)}
                stream.write(json.dumps(value, separators=(",", ":")) + "\n")
                users[key] += 1; pairs[_canonical([key, target]).decode()] += 1; files[row.file_index] += 1
                stream_digest.update(_canonical([key, list(row.items), target]) + b"\n")
                rows_count += 1; item_count += len(row.items)
                low, high = min(row.items), max(row.items)
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                shortest_history = len(row.items) if shortest_history is None else min(shortest_history, len(row.items))
                longest_history = len(row.items) if longest_history is None else max(longest_history, len(row.items))
                if first_truncated is None and len(row.items) * 4 > lc.sequence_length:
                    first_truncated = {**value, "selection_rule": "First testing record longer than the fixed SID-token window, before any predictions",
                        "input_ids": tensors[1].tolist(), "attention_mask": tensors[2].tolist(),
                        "visible_history_item_ids": tensors[4]}
        for row in _iter_metadata(metadata["file_order"]):
            _validate_items(row, sid.shape[1])
            pending.append(row)
            if len(pending) == 512:
                record_batch(pending); pending = []
        if pending:
            record_batch(pending)
    if rows_count == 0:
        raise ValueError("Testing metadata contains no records")
    _sources_unchanged(metadata)
    repeated = {key: count for key, count in users.items() if count > 1}
    result = {"schema": "beauty-full-test-coverage-v1", "split": "testing",
        "rows": rows_count, "targets": rows_count, "unique_users": len(users),
        "duplicate_user_rows": rows_count - len(users), "duplicate_user_keys": repeated,
        "total_raw_item_occurrences": item_count, "item_range": [minimum, maximum],
        "raw_items_per_record_range": [shortest_history, longest_history],
        "first_truncated_engineering_example": first_truncated,
        "sid_item_count": sid.shape[1], "sid": dict(spec["sid"]), "sid_dtype": str(sid.dtype),
        "key_set_sha256": hashlib.sha256(_canonical(sorted(users))).hexdigest(),
        "key_target_multiset_sha256": hashlib.sha256(_canonical(sorted(pairs.items()))).hexdigest(),
        "ordered_metadata_sha256": stream_digest.hexdigest(), "loader": metadata,
        "rows_per_file": [{"source_file_index": index, "rows": files[index]} for index in range(len(metadata["file_order"]))],
        "target_index": str(index_path), "target_index_sha256": _hash(index_path),
        "target_protocol": "Verified SID at original raw last item; actual NextKTokenMasking(next_k=4) executed for every row in CPU metadata chunks <=512",
        "original_label_targets_verified": True, "complete_target_hidden_groups_verified": True,
        "metadata_chunk_size": 512, "metadata_chunk_is_model_batch": False,
        "raw_user_ids_exported": False, "model_execution": False,
        "coverage_is_not_model_evaluation": True}
    if sum(value["rows"] for value in result["rows_per_file"]) != rows_count:
        raise ValueError("Per-file testing metadata counts do not sum to total rows")
    _write_new(output / "test_coverage.json", result)
    return result
