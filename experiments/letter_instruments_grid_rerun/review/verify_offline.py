"""Independent replay against the delivered preview, using only Python's standard library."""
import argparse
from collections import Counter, defaultdict
from gzip import open as gzip_open
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import struct
from zipfile import ZipFile


def load(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_i64(path, count):
    """Read the inspected contiguous little-endian integer storage without loading pickle."""
    with ZipFile(path) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith("/data/0")]
        assert archive.read(name.split("/")[0] + "/byteorder") == b"little"
        return struct.unpack(f"<{count}q", archive.read(name))


def canonical_catalog(run):
    upper = run / "upstream"
    table = rows(upper / "item_sids.jsonl")
    codes = [r["sid"] for r in table]
    assert len(table) == 9922 and [r["item_id"] for r in table] == list(range(9922))
    assert all(len(row) == 4 and all(type(x) is int and 0 <= x < 256 for x in row) for row in codes)
    assert len(set(map(tuple, codes))) == 9922
    sid_file = upper / "semantic_ids.pt"
    assert archive_i64(sid_file, 4 * 9922) == tuple(x for level in zip(*codes) for x in level)
    predicted = upper / "sid_generation/predicted_codes_by_item.pt"
    assert archive_i64(predicted, 3 * 9922) == tuple(x for sid in codes for x in sid[:3])
    groups = defaultdict(list)
    for item, sid in enumerate(codes):
        groups[tuple(sid[:3])].append(item)
    for items in groups.values():
        for serial, item in enumerate(items, start=1):
            assert codes[item][3] == (serial if len(items) > 1 else 0)
    return codes, {"sha256": sha(sid_file), "prefixes": len(groups),
                   "unique_full_codes": len(codes), "fourth_max": max(row[3] for row in codes)}


def varints(blob):
    value = shift = 0
    for byte in blob:
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            assert shift < 70
        else:
            yield value
            value = shift = 0
    assert shift == 0


def message_fields(blob):
    pos = 0
    while pos < len(blob):
        tag = blob[pos]
        assert tag & 7 == 2 and tag < 128
        pos += 1
        size = next(varints(blob[pos:pos + 10]))
        pos += max(1, (size.bit_length() + 6) // 7)
        assert pos + size <= len(blob)
        yield tag >> 3, blob[pos:pos + size]
        pos += size


def only(blob, tag):
    ((actual, child),) = message_fields(blob)
    assert actual == tag
    return child


def crc_lookup(value):
    for _ in range(8):
        value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value


CRC32C_TABLE = tuple(crc_lookup(x) for x in range(256))


def masked_crc(blob):
    state = 0xFFFFFFFF
    for byte in blob:
        state = (state >> 8) ^ CRC32C_TABLE[(state ^ byte) & 255]
    crc = state ^ 0xFFFFFFFF
    rotate = (crc >> 15) | (crc << 17)
    return (rotate + 0xA282EAD8) & 0xFFFFFFFF


def verify_splits(root, users):
    assert tuple(users) == tuple(str(i) for i in range(24772))
    summary = {}
    for split in ("training", "evaluation", "testing"):
        path = root / "data" / split / "part-00000.tfrecord.gz"
        stored = 0
        with gzip_open(path, "rb") as data:
            for i, sequence in enumerate(users.values()):
                width_bytes = data.read(8)
                assert len(width_bytes) == 8
                (width,) = struct.unpack("<Q", width_bytes)
                assert width <= 100_000 and struct.unpack("<I", data.read(4))[0] == masked_crc(width_bytes)
                frame = data.read(width)
                assert len(frame) == width and struct.unpack("<I", data.read(4))[0] == masked_crc(frame)
                values = {}
                for feature_tag, feature in message_fields(only(frame, 1)):
                    assert feature_tag == 1
                    entry = dict(message_fields(feature))
                    values[entry[1].decode()] = list(varints(only(only(entry[2], 3), 1)))
                expected = {"training": sequence[:-2], "evaluation": sequence[-22:-1],
                            "testing": sequence[-21:]}[split]
                assert values == {"user_id": [i], "sequence_data": expected}
                stored += len(expected)
            assert not data.read(1)
        summary[split] = {"users": len(users), "stored_items": stored, "sha256": sha(path)}
    return summary


def metrics_for(row, target):
    probabilities = struct.unpack("<10f", bytes.fromhex(row["scores_f32le"]))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
    candidates = row["ids"]
    assert len(candidates) == 10 and all(len(sid) == 4 for sid in candidates)
    order = sorted(range(10), key=lambda j: (-probabilities[j], j))
    hits = [i + 1 for i, j in enumerate(order) if candidates[j] == target]
    first = hits[0] if hits else 0
    native = {}
    unified = {"hit@1": float(first == 1)}
    for k in (5, 10):
        for metric in ("recall", "ndcg"):
            name = f"{metric}@{k}"
            top = row["native_topk"][name]
            assert len(top) == len(set(top)) == k
            assert [probabilities[j] for j in top] == sorted(probabilities, reverse=True)[:k]
            positions = [i + 1 for i, j in enumerate(top) if candidates[j] == target]
            denom = min(k, len(hits))
            if not denom:
                native[name] = 0.0
            elif metric == "recall":
                native[name] = len(positions) / denom
            else:
                native[name] = math.fsum(1 / math.log2(i + 1) for i in positions) / math.fsum(
                    1 / math.log2(i + 1) for i in range(1, denom + 1)
                )
        qualified = 0 < first <= k
        unified[f"hit@{k}"] = float(qualified)
        unified[f"mrr@{k}"] = 1 / first if qualified else 0.0
        unified[f"ndcg@{k}"] = math.fsum(1 / math.log2(i + 1) for i in hits if i <= k)
        # This catalogue has unique SIDs. Non-catalogue SIDs keep their rank slot.
        unified[f"ItemHit@{k}"] = float(qualified)
        unified[f"ItemNDCG@{k}"] = 1 / math.log2(first + 1) if qualified else 0.0
    return {"native": native, "unified": unified}


def compare_candidates(run, users, codes, projection):
    catalogue = {tuple(sid) for sid in codes}
    prefix = run / "evaluation"
    axis = {entry["user_id"]: entry for entry in rows(projection)}
    normal = rows(prefix / "Normal/candidates.jsonl")
    assert (prefix / "Normal/candidates.jsonl").read_bytes() == (prefix / "Normal/native_shadow.jsonl").read_bytes()
    summaries = {}
    best = load(run / "training/best.json")
    best_path = run / "training" / best["file"]
    assert sha(best_path) == best["sha256"]
    for mode, active in (("Normal", 4), ("Shuffle3a", 3), ("Shuffle4", 4)):
        path = prefix / mode
        manifest = load(path / "native_evaluation.json")
        assert manifest["reload"]["strict"] and manifest["reload"]["checkpoint_sha256"] == best["sha256"]
        assert manifest["checks"]["batch_size_8"] == 2 and manifest["checks"]["batch_size_4"] == 1
        observed = rows(path / "candidates.jsonl")
        assert len(observed) == 20 and len({r["user_id"] for r in observed}) == 20
        counters = Counter()
        averages = {"native": Counter(), "unified": Counter()}
        for reference, record in zip(normal, observed, strict=True):
            uid = record["user_id"]
            assert uid == reference["user_id"] and record["mode"] == mode and record["global_row"] == int(uid)
            history, target_item = users[uid][-21:-1], users[uid][-1]
            target = codes[target_item]
            assert record["target_item_id"] == target_item and record["target_sid"] == target
            assert record["history_item_ids"] == history
            original = [v for item in history for v in codes[item]]
            padding = 120 - len(original)
            assert record["input_ids"] == original + [-1] * padding
            assert record["input_mask"] == [1] * len(original) + [0] * padding
            before = [v + 256 * (i % 4) for i, v in enumerate(original)]
            choices = [p + tuple(range(active, 4)) for p in permutations(range(active))]
            picks = [0] * len(history) if mode == "Normal" else axis[uid]["a0_and_a1_permutation_ids"][mode.lower()]
            pairs = zip(picks, record["draws"], strict=True)
            after = []
            for j, (index, (permutation, observed_index, rejected)) in enumerate(pairs):
                assert list(choices[index]) == permutation and index == observed_index and rejected == 0
                group = before[4 * j:4 * j + 4]
                moved = [group[k] for k in choices[index]]
                assert sorted(moved) == group and (active == 4 or group[3] == moved[3])
                after.extend(moved)
                counters["nonidentity_draws"] += int(index != 0)
                counters["token_changes"] += int(moved != group)
            assert record["offset_ids_before"] == before + [0] * padding
            assert record["offset_ids_after"] == after + [0] * padding
            values = metrics_for(record, target)
            for scope in values:
                averages[scope].update(values[scope])
            candidates = [tuple(c) for c in record["ids"]]
            counters["candidate_ids_changed"] += int(reference["ids"] != record["ids"])
            counters["candidate_scores_changed"] += int(reference["scores_f32le"] != record["scores_f32le"])
            counters["out_of_catalog"] += sum(c not in catalogue for c in candidates)
            counters["duplicate_sids"] += len(candidates) - len(set(candidates))
        means = {g: {k: v / len(observed) for k, v in m.items()} for g, m in averages.items()}
        offline = load(path / "offline_evaluation.json")
        for group, metrics in offline["metrics"].items():
            for metric, expected in metrics.items():
                assert abs(expected - means[group][metric]) < 1e-12
        assert offline["coverage"] == len(observed) and offline["counts"]["candidates"] == 200
        assert all(offline["counts"][k] == counters[k] for k in ("duplicate_sids", "out_of_catalog"))
        summaries[mode] = {"users": len(observed), "candidates": 200, "metrics": means, "counts": dict(counters)}
    return summaries


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--job", type=Path, required=True)
    cli.add_argument("--inter", type=Path, required=True)
    cli.add_argument("--prior-v2", type=Path, required=True)
    args = cli.parse_args()
    run = args.job / "output/run"
    users = load(args.inter)
    assert sum(map(len, users.values())) == 206153
    codes, sid = canonical_catalog(run)
    bound = run / "bound/codex/letter_official_downstream_20261116/inputs/instruments"
    assert sha(bound / "semantic_ids.pt") == sid["sha256"]
    status = load(args.job / "status.json")
    assert (status["status"], status["exit_code"], status["cleanup"]["verified"]) == ("SUCCEEDED", 0, True)
    result = {"job": status["job_id"], "sid": sid, "tfrecords": verify_splits(bound, users),
              "evaluation": compare_candidates(run, users, codes, args.prior_v2)}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
