"""Prepare and independently verify the fixed Instruments axis and native TFRecords."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE / "historical"
RAW = Path("third_party/LETTER-8d0154e/data/Instruments")
PREPARED = Path("codex/letter_official_downstream_20261116/inputs/instruments")
SPLITS = ("training", "evaluation", "testing")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha(f)


def _sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def official(root):
    historical = load(HISTORICAL / "mapping_audit.json")
    hashes = {Path(key).name: value for key, value in historical["official_sources"].items()}
    files = {kind: root / RAW / f"Instruments.{kind}.json" for kind in ("index", "inter", "item")}
    for path in files.values():
        assert sha(path) == hashes[path.name], path
    ids, users, items = (load(files[k]) for k in ("index", "inter", "item"))
    assert set(ids) == set(items) == {str(i) for i in range(9922)}
    assert set(users) == {str(i) for i in range(24772)}
    codes = []
    for i in range(len(items)):
        tokens = ids[str(i)]
        assert len(tokens) == 4
        row = tuple(int(re.fullmatch(f"<{level}_(\\d+)>", token).group(1))
                    for level, token in zip("abcd", tokens))
        assert all(0 <= code < 256 for code in row)
        codes.append(row)
    seqs = [users[str(i)] for i in range(len(users))]
    assert {item for seq in seqs for item in seq} == set(range(len(items)))
    assert sum(map(len, seqs)) == 206153
    assert not any(len(set(seq)) != len(seq) for seq in seqs)
    groups = Counter(codes)
    assert (len(groups), sum(n > 1 for n in groups.values()), sum(n for n in groups.values() if n > 1)) == (9897, 23, 48)
    assert sum(groups[codes[seq[-1]]] > 1 for seq in seqs) == 122
    return codes, seqs, groups


def split_seq(seq, split):
    return {"training": seq[:-2], "evaluation": seq[:-1][-21:], "testing": seq[-21:]}[split]


def varint(value, pos):
    result = 0
    for shift in range(0, 70, 7):
        byte = value[pos]
        pos += 1
        result |= (byte & 127) << shift
        if byte < 128:
            return result, pos
    raise ValueError("Malformed protobuf varint")


def fields(value):
    pos = 0
    while pos < len(value):
        tag, pos = varint(value, pos)
        assert tag & 7 == 2
        length, pos = varint(value, pos)
        assert pos + length <= len(value)
        yield tag >> 3, value[pos:pos + length]
        pos += length


def only(value, field):
    ((tag, part),) = fields(value)
    assert tag == field
    return part


def record_fields(payload):
    """Decode the two int64-list features from a tf.train.Example."""
    result = {}
    for tag, entry in fields(only(payload, 1)):
        assert tag == 1
        entries = dict(fields(entry))
        assert set(entries) == {1, 2}
        key = entries[1].decode("ascii")
        packed = only(only(entries[2], 3), 1)
        values = []
        pos = 0
        while pos < len(packed):
            number, pos = varint(packed, pos)
            values.append(number)
        assert key not in result
        result[key] = values
    assert set(result) == {"user_id", "sequence_data"}
    return result


def records(path):
    # SHA-256 checks the exact artifacts. gzip also verifies the container CRC.
    with gzip.open(path, "rb") as reader:
        while header := reader.read(12):
            assert len(header) == 12
            length = struct.unpack("<Q", header[:8])[0]
            assert length <= 100_000
            payload = reader.read(length)
            footer = reader.read(4)
            assert len(payload) == length and len(footer) == 4
            yield record_fields(payload), header + payload + footer


def sample_users(codes, seqs, groups):
    """Freeze before model initialization. Cover collision, history extremes and a 4-row tail."""
    shortest = min(range(len(seqs)), key=lambda i: len(seqs[i]))
    longest = next(i for i, seq in enumerate(seqs) if len(seq) >= 21)
    collision = next(i for i, seq in enumerate(seqs) if groups[codes[seq[-1]]] > 1)
    selected = set(range(20 - len({shortest, longest, collision, len(seqs) - 1})))
    selected.update((shortest, longest, collision, len(seqs) - 1))
    for i in range(len(seqs)):
        if len(selected) == 20:
            break
        selected.add(i)
    return sorted(selected)


def verify(root):
    codes, seqs, groups = official(root)
    directory = root / PREPARED
    audit = load(directory / "mapping_audit.json")
    builds = load(directory / "preparation.json")
    tensor_path = directory / "semantic_ids.pt"
    assert builds["sid_torch_readback_exact"] and sha(tensor_path) == audit["sid_sha256"]
    # Compare all stored int64 values without importing Torch or deserializing pickle.
    with ZipFile(tensor_path) as archive:
        assert archive.testzip() is None
        (storage,) = [name for name in archive.namelist() if name.endswith("/data/0")]
        assert archive.read(storage.split("/")[0] + "/byteorder") == b"little"
        assert archive.read(storage) == struct.pack(f"<{4 * len(codes)}q", *(i for col in zip(*codes) for i in col))
    expected_sample = sample_users(codes, seqs, groups)
    assert builds["preflight_test_users"] == expected_sample
    checked = {}
    for split in SPLITS:
        path = directory / "data" / split / "part-00000.tfrecord.gz"
        assert sha(path) == audit["data_files"][str(path.relative_to(root))]
        rows = 0
        for row, _ in records(path):
            assert row["user_id"] == [rows]
            assert row["sequence_data"] == split_seq(seqs[rows], split)
            rows += 1
        assert rows == len(seqs)
        checked[split] = rows
    subset = directory / "preflight_testing/part-00000.tfrecord.gz"
    assert sha(subset) == builds["preflight_test_sha256"]
    subset_rows = [(row["user_id"][0], row["sequence_data"]) for row, _ in records(subset)]
    assert subset_rows == [(i, seqs[i][-21:]) for i in expected_sample]
    return {"items": len(codes), "rows_exact": checked, "preflight_users": expected_sample,
            "interaction_count": sum(map(len, seqs)), "unique_sids": len(groups),
            "colliding_groups": sum(n > 1 for n in groups.values()),
            "colliding_items": sum(n for n in groups.values() if n > 1),
            "ambiguous_test_targets": sum(groups[codes[seq[-1]]] > 1 for seq in seqs)}


def prepare(root):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(HERE / "native"))
    from codex.letter_official_downstream_20261116 import prepare_data
    import torch
    from shutil import copyfile

    codes, seqs, groups = official(root)
    upstream = root / "codex/letter_upstream_transformer_20261115"
    upstream.mkdir(parents=True, exist_ok=True)
    copyfile(HISTORICAL / "official_source_manifest.json", upstream / "source_manifest.json")
    prepare_data.ROOT = root
    prepare_data.HERE = root / PREPARED.parents[1]
    prepare_data.AUTHOR = root / RAW.parents[1]
    prepare_data.prepare("instruments")
    directory = root / PREPARED
    tensor = torch.load(directory / "semantic_ids.pt", weights_only=True)
    assert tensor.dtype == torch.long and tensor.shape == (4, 9922)
    assert torch.equal(tensor, torch.tensor(codes).T)
    audit = load(directory / "mapping_audit.json")
    reference = load(HISTORICAL / "mapping_audit.json")
    assert audit["splits"] == reference["splits"]
    selected = sample_users(codes, seqs, groups)
    subset = directory / "preflight_testing/part-00000.tfrecord.gz"
    subset.parent.mkdir()
    with gzip.open(subset, "wb") as output:
        for row, frame in records(directory / "data/testing/part-00000.tfrecord.gz"):
            if row["user_id"][0] in selected:
                output.write(frame)
    write(directory / "preparation.json", {
        "sid_torch_readback_exact": True, "sid_sha256_equals_historical": audit["sid_sha256"] == reference["sid_sha256"],
        "split_sha256_equals_historical": {p: h == reference["data_files"][p] for p, h in audit["data_files"].items()},
        "preflight_test_users": selected, "preflight_test_sha256": sha(subset)})
    return verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps((prepare if args.mode == "prepare" else verify)(args.root.resolve()), indent=2))
