"""Freeze and verify the common LETTER Instruments catalog and its reused GRID vectors."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from zipfile import ZipFile


LETTER_COMMIT = "8d0154e28de37dbb6e24871c508ad8ddb1921cda"
GRID_COMMIT = "bc6176dc144b827b1b29f73ad3b216e28b39f48f"
SHUFFLE_COMMIT = "af360a158280165f07249b80d2cfa0e871902498"
LETTER_ITEMS = 9922
LETTER_USERS = 24772
GRID_ITEMS = 10611
EMBEDDING_WIDTH = 3584
ASIN = re.compile(r"[0-9A-Z]{10}")
LETTER_FILES = {
    "Instruments.inter.json": "0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743",
    "Instruments.item.json": "1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2",
    "Instruments.index.json": "33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd",
}
LC_ARCHIVE_SHA256 = "96aad0cf1496a8ffef31bf1ca3a0d1033ca49473dff46b2c2dc27763224f9453"
LC_MAP_SHA256 = "76380000fb809cf19e841c69163310ec2d32b7b578277e2b659fbd1e72714688"
GRID_ARCHIVE_SHA256 = "8801dfe5d3fcc75f99101dea1d50cf5abc742d44ce3771746ffd04ff9277929a"
GRID_FILES = {
    "item_mapping.jsonl": "262c3bc796ece5826f2e171f585c1664750e9d5b3ef336b5d13899f9715246e8",
    "items_qwen.jsonl": "2d3bcabe02e0656958fc2269e30308813377413de4a35a7202d17808f634e0f1",
    "item_embeddings.pt": "65adf94502f35b93fb48964d022f84f2afb7f512282946092ad5202d11baa165",
    "grid_assets.json": "727b488ed7e95e3a68cc0331148983605e4539b7a187ddc56e0883651f02d64a",
    "grid_item_embeddings.json": "1bf1349803776becc00cd07d76016038fd4f8620b6702856e7961ede9be719bd",
    "grid_qwen_config.json": "78419775a22e97926283b8027331aeafdddbffe3f4cf413b4096fb71fd74c32f",
}
REPLAY_FILES = {
    "a0": "3acc7910efbbdc86b2676d0c7f383272bd062bbb12a207f09f5b8a73cd00238d",
    "a1": "93e7a63e53715e639b7d6aea088b4d68d1309658e7a9a8b04cddfbbc0d99d31c",
}
AXIS_SHA256 = "1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def check_hashes(root, expected):
    for filename, digest in expected.items():
        path = root / filename
        require(sha(path) == digest, f"SHA-256 mismatch: {path}")


def unique_keys(pairs):
    mapping = {}
    for key, value in pairs:
        require(key not in mapping, f"duplicate JSON key: {key}")
        mapping[key] = value
    return mapping


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line, object_pairs_hook=unique_keys)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def write_jsonl(path, rows):
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def letter_axis(directory):
    check_hashes(directory, LETTER_FILES)
    users = read_json(directory / "Instruments.inter.json")
    items = read_json(directory / "Instruments.item.json")
    indices = read_json(directory / "Instruments.index.json")
    expected_items = {str(item) for item in range(LETTER_ITEMS)}
    require(set(items) == set(indices) == expected_items, "LETTER item axis mismatch")
    require(list(users) == [str(user) for user in range(LETTER_USERS)], "LETTER user axis mismatch")
    item_degree = Counter()
    for user, sequence in users.items():
        require(len(sequence) >= 5 and len(set(sequence)) == len(sequence), f"invalid history: {user}")
        require(all(type(item) is int and 0 <= item < LETTER_ITEMS for item in sequence), f"bad item for {user}")
        item_degree.update(sequence)
    require(len(item_degree) == LETTER_ITEMS, "incomplete item coverage")
    require(sum(item_degree.values()) == 206153 and min(item_degree.values()) == 5, "unexpected LETTER data counts")
    return users, indices, item_degree


def item2id(archive_path, letter_dir):
    require(sha(archive_path) == LC_ARCHIVE_SHA256, "unexpected LC-Rec archive")
    with ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "LC-Rec archive CRC mismatch")
        for filename in ("Instruments.inter.json", "Instruments.item.json"):
            member = archive.read(f"Instruments/{filename}")
            require(member == (letter_dir / filename).read_bytes(), f"LC-Rec and LETTER differ for {filename}")
        content = archive.read("Instruments/Instruments.item2id")
    require(digest_bytes(content) == LC_MAP_SHA256, "unexpected original item2id")
    mapping = {}
    for line in content.decode("ascii").splitlines():
        asin, value = line.split("\t")
        item = int(value)
        require(ASIN.fullmatch(asin) and item not in mapping, f"invalid LC-Rec item2id row: {line}")
        mapping[item] = asin
    require(set(mapping) == set(range(LETTER_ITEMS)) and len(set(mapping.values())) == LETTER_ITEMS,
            "LC-Rec item2id is not bijective over LETTER catalog")
    return mapping, content


def validate_grid_sources(source):
    check_hashes(source, GRID_FILES)
    archive = read_json(source / "grid_assets.json")["assets"]["instruments-grid-original-assets.zip"]
    require(archive["sha256"] == GRID_ARCHIVE_SHA256, "wrong GRID release archive")
    member_info = {Path(row["member"]).name: row for row in archive["members"]}
    for name in ("item_mapping.jsonl", "items_qwen.jsonl", "item_embeddings.pt"):
        entry = member_info[name]
        require(entry["sha256"] == GRID_FILES[name] and entry["bytes"] == (source / name).stat().st_size,
                f"archive member mismatch: {name}")
    metadata = read_json(source / "grid_item_embeddings.json")
    qwen_config = read_json(source / "grid_qwen_config.json")
    require(metadata["complete"] and metadata["shape"] == [GRID_ITEMS, EMBEDDING_WIDTH]
            and metadata["stored_dtype"] == "float32"
            and metadata["sha256"] == GRID_FILES["item_embeddings.pt"]
            and metadata["source_sha256"] == qwen_config["input_sha256"] == GRID_FILES["items_qwen.jsonl"],
            "GRID embedding metadata disagrees with text or embedding identity")


def old_grid_axis(source):
    item_rows = list(read_jsonl(source / "item_mapping.jsonl"))
    text_rows = list(read_jsonl(source / "items_qwen.jsonl"))
    expected = list(range(GRID_ITEMS))
    require([row["item_id"] for row in item_rows] == expected, "old GRID ID order mismatch")
    require([row["item_id"] for row in text_rows] == expected, "old Qwen text/embedding row order mismatch")
    by_asin = {}
    texts = []
    for identity, text in zip(item_rows, text_rows, strict=True):
        asin = identity["asin"]
        require(ASIN.fullmatch(asin) and asin not in by_asin, f"non-unique or invalid GRID ASIN: {asin}")
        require(type(text["text"]) is str and text["text"], f"missing original Qwen text: {identity['item_id']}")
        by_asin[asin] = identity["item_id"]
        texts.append(text["text"])
    return by_asin, texts


def vector_sha(row):
    # Canonical hash uses exact original FP32 bytes, including signed zeros.
    return digest_bytes(row.numpy().astype("<f4", copy=False).tobytes(order="C"))


def bind_catalog(asin_by_item, old_by_asin, texts, source, output):
    import torch

    if missing := sorted(set(asin_by_item.values()) - set(old_by_asin)):
        raise ValueError(f"{len(missing)} unmatched LETTER ASINs: {missing}")
    old_rows = [old_by_asin[asin_by_item[item]] for item in range(LETTER_ITEMS)]
    embedding = torch.load(source / "item_embeddings.pt", map_location="cpu", weights_only=True)
    require(isinstance(embedding, torch.Tensor) and embedding.dtype == torch.float32
            and tuple(embedding.shape) == (GRID_ITEMS, EMBEDDING_WIDTH)
            and torch.isfinite(embedding).all().item(), "invalid original embedding tensor")
    selected = embedding.index_select(0, torch.tensor(old_rows, dtype=torch.long))
    path = output / "item_embeddings.pt"
    with path.open("xb") as file:
        torch.save(selected, file)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    require(tuple(reloaded.shape) == (LETTER_ITEMS, EMBEDDING_WIDTH) and reloaded.dtype == embedding.dtype
            and torch.isfinite(reloaded).all().item(), "invalid reloaded common embedding tensor")
    bridge = []
    ordered_text = []
    for item, old_item in enumerate(old_rows):
        row_digest = vector_sha(embedding[old_item])
        require(vector_sha(reloaded[item]) == row_digest, f"changed FP32 row {item}")
        text = texts[old_item]
        bridge.append({"item_id": item, "letter_item_id": item, "asin": asin_by_item[item],
                       "old_grid_item_id": old_item, "old_grid_embedding_row": old_item,
                       "text_sha256": digest_bytes(text.encode("utf-8")), "embedding_row_sha256": row_digest})
        ordered_text.append({"item_id": item, "text": text})
    write_jsonl(output / "item_identity.jsonl", bridge)
    write_jsonl(output / "items_qwen.jsonl", ordered_text)
    return bridge


def check_sequences(users, bridge, item_degree):
    letter_to_asin = {row["letter_item_id"]: row["asin"] for row in bridge}
    asin_to_item = {row["asin"]: row["item_id"] for row in bridge}
    item_to_letter = {row["item_id"]: row["letter_item_id"] for row in bridge}
    axis = {}
    length_counts = Counter()
    split_counts = Counter()
    full_sequence_hash = hashlib.sha256()
    asin_sequence_hash = hashlib.sha256()
    for ordinal, (user, sequence) in enumerate(users.items()):
        axis[user] = ordinal
        asin_seq = [letter_to_asin[item] for item in sequence]
        roundtrip = [item_to_letter[asin_to_item[asin]] for asin in asin_seq]
        require(roundtrip == sequence, f"sequence roundtrip mismatch: {user}")
        length_counts[len(sequence)] += 1
        split_counts["train_saved_items"] += len(sequence) - 2
        split_counts["validation_visible_history_items"] += min(20, len(sequence) - 2)
        split_counts["test_visible_history_items"] += min(20, len(sequence) - 1)
        split_counts["validation_targets"] += 1
        split_counts["test_targets"] += 1
        full_sequence_hash.update(canonical([user, sequence]) + b"\n")
        asin_sequence_hash.update(canonical([user, asin_seq]) + b"\n")
    require(digest_bytes(canonical(list(axis))) == AXIS_SHA256, "original V2 evaluation user axis mismatch")
    return axis, {
        "users": len(users), "items": len(bridge), "interactions": sum(item_degree.values()),
        "min_user_interactions": min(length_counts), "min_item_interactions": min(item_degree.values()),
        "user_sequence_lengths": dict(sorted(length_counts.items())), "duplicate_user_item_rows": 0,
        "sequence_roundtrip_users": len(users), "ordered_user_id_sequence_sha256": full_sequence_hash.hexdigest(),
        "ordered_user_asin_sequence_sha256": asin_sequence_hash.hexdigest(),
        "ordered_user_id_axis_sha256": AXIS_SHA256, "split_counts": dict(split_counts),
        "split_definition": {"train": "S[:-2]", "validation": "S[:-1][-21:]", "test": "S[-21:]"},
    }


def replay_projection(users, indices, a0_path, a1_path, output):
    paths = {"a0": a0_path, "a1": a1_path}
    for route, path in paths.items():
        require(sha(path) == REPLAY_FILES[route], f"modified original V2 replay file: {route}")
    counts = {"shuffle3a": Counter(), "shuffle4": Counter()}
    occurrences = 0
    with (output / "prior_v2_replay_projection.jsonl").open("xb") as stream:
        for (user_id, sequence), a0, a1 in zip(users.items(), read_jsonl(a0_path), read_jsonl(a1_path), strict=True):
            visible = sequence[:-1][-20:]
            target = sequence[-1]
            expected = (int(user_id), user_id, "".join(indices[str(target)]), len(visible))
            for route, row in (("a0", a0), ("a1", a1)):
                actual = tuple(row[key] for key in ("row_idx", "user_id", "target", "visible_history_length"))
                require(actual == expected, f"{route} V2 row/target/history length mismatch: {user_id}")
            require(a1["target_item_id"] == str(target), f"A1 item target mismatch: {user_id}")
            draws = {}
            for mode, size in (("shuffle3a", 6), ("shuffle4", 24)):
                source_draws = a0["conditions"][mode]["permutation_ids"]
                require(source_draws == a1["conditions"][mode]["permutation_ids"]
                        and len(source_draws) == len(visible)
                        and all(type(n) is int and 0 <= n < size for n in source_draws),
                        f"A0/A1 {mode} draw mismatch: {user_id}")
                draws[mode] = source_draws
                counts[mode].update(source_draws)
            record = {"row_idx": int(user_id), "user_id": user_id, "target_item_id": target,
                      "historical_letter_target_sid": expected[2], "visible_letter_ids": visible,
                      "a0_and_a1_permutation_ids": draws}
            stream.write(canonical(record) + b"\n")
            occurrences += len(visible)
    return {"users": len(users), "visible_occurrences": occurrences,
            "permutation_counts": {mode: dict(sorted(count.items())) for mode, count in counts.items()},
            "source_replay_paths": {route: str(path) for route, path in paths.items()},
            "source_replay_sha256": REPLAY_FILES}


def build(args):
    require(sys.byteorder == "little", "expected little-endian deployment")
    users, indices, by_degree = letter_axis(args.letter)
    asin_by_item, original_map = item2id(args.lc_archive, args.letter)
    validate_grid_sources(args.grid)
    old_by_asin, texts = old_grid_axis(args.grid)
    output = args.output
    staging = output.with_name(output.name + ".building")
    require(not output.exists(), f"output already exists: {output}")
    staging.mkdir()
    for name in LETTER_FILES:
        shutil.copyfile(args.letter / name, staging / name)
    (staging / "Instruments.item2id").write_bytes(original_map)
    bridge = bind_catalog(asin_by_item, old_by_asin, texts, args.grid, staging)
    axis, data_report = check_sequences(users, bridge, by_degree)
    write_json(staging / "user_axis.json", axis)
    replay_report = replay_projection(users, indices, args.a0_replay, args.a1_replay, staging)
    require(replay_report["visible_occurrences"] == data_report["split_counts"]["test_visible_history_items"],
            "prior V2 and planned test history occurrences disagree")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(staging.iterdir())}
    write_json(staging / "assets_manifest.json", {
        "schema": 1, "letter_commit": LETTER_COMMIT, "grid_commit": GRID_COMMIT,
        "shuffle_commit": SHUFFLE_COMMIT, "build_argv": sys.argv,
        "letter_files": LETTER_FILES, "lc_archive_sha256": LC_ARCHIVE_SHA256,
        "lc_archive_source": "https://drive.google.com/file/d/19tsIxV_LbErh5FfYhT732M82y0V-3bxR/view",
        "lc_archive_matches_letter": ["Instruments.inter.json", "Instruments.item.json"],
        "lc_archive_sid_index_consumed": False, "grid_archive_sha256": GRID_ARCHIVE_SHA256,
        "grid_sources": GRID_FILES, "grid_items": GRID_ITEMS, "shared_items": len(bridge),
        "letter_only_items": 0, "grid_only_items": GRID_ITEMS - len(bridge),
        "full_grid_fp32_shape": [GRID_ITEMS, EMBEDDING_WIDTH], "embedding_shape": [len(bridge), EMBEDDING_WIDTH],
        "embedding_dtype": "float32", "row_hash_encoding": "contiguous little-endian IEEE 754 float32 bytes",
        "embedding_bit_exact_reloaded_rows": len(bridge), "embedding_nonfinite_values": 0,
        "historical_qwen_revision": "Historical archive does not verify a unique HF revision for all weight shards.",
        "data": data_report, "prior_v2": replay_report, "files": files,
    })
    verify(staging, sha(staging / "assets_manifest.json"))
    staging.rename(output)
    print(f"Verified common inputs at {output}")
    print(f"assets_manifest.json SHA-256 {sha(output / 'assets_manifest.json')}")


def verify(root, manifest_sha256):
    manifest_path = root / "assets_manifest.json"
    require(sha(manifest_path) == manifest_sha256, "common asset manifest differs from pinned identity")
    manifest = read_json(manifest_path)
    require(manifest["schema"] == 1 and manifest["letter_commit"] == LETTER_COMMIT
            and manifest["grid_commit"] == GRID_COMMIT and manifest["shared_items"] == LETTER_ITEMS,
            "different common data lineage")
    for filename, entry in manifest["files"].items():
        require(Path(filename).name == filename, f"unexpected asset path: {filename}")
        path = root / filename
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"]
                and sha(path) == entry["sha256"], f"common asset differs from pinned identity: {filename}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    for arg in ("letter", "lc-archive", "grid", "a0-replay", "a1-replay", "output"):
        build_parser.add_argument("--" + arg, type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest-sha256", required=True)
    args = cli.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args.root, args.manifest_sha256)


if __name__ == "__main__":
    main()
