"""Content identity for the original LETTER Beauty inputs and new checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

HISTORICAL_COMMIT = "6e1b211d9c37f358c43b1540b40b17f6703150d2"
HISTORICAL_CONFIG_SHA = "24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd"
MODEL_REVISION = "4782ad278652c7c71b72204d462d6d01eaaf7549"
DATASET = "Beauty"
EXPERIMENT_ID = "letter-beauty-llama-history-attention-seed42"
MODES = {"bidirectional", "causal"}
MODEL_FILES = {
    "config.json": "e4c3dea29aefc16c0e63825f9ae097dbf5cb5d7c320843a84257e11be3b452c8",
    "model.safetensors.index.json": "217d527037a95cd5cd30384afb425987824373b94cc7206beb48ec0c9499c98f",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "tokenizer_config.json": "fc6996ba04b235235cf9c7bd00fd8817c9b8abf445fe14b0a36ef900e3b9cfe8",
    "special_tokens_map.json": "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531",
    "generation_config.json": "fd7ff399e5568cc21a0a8414f43df88ef7c424995b9b97a90563165d2cf79efd",
}
# Preserve the identity of the previously verified 13.5 GB payload. A file
# whose size or historical modification time has changed must be rehashed.
MODEL_SHARDS = {
    "model-00001-of-00002.safetensors": (
        9976579217, 1794238862607969478,
        "d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12",
    ),
    "model-00002-of-00002.safetensors": (
        3500297444, 1794243142341035385,
        "94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee",
    ),
}
ASSETS = {
    "index": "be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82",
    "inter": "8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5",
    "item": "74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330",
}
NEW_TOKEN_COUNT = 965
VOCAB_SIZE = 32965
TRAINABLE_PARAMETERS = 290037760
EXECUTION_TOPOLOGY = "SINGLE_GPU"
EXECUTION_PROFILE = {
    "world_size": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "global_batch": 64,
}
ADAPTER_MEMBERS = {
    "adapter_model.bin", "adapter_config.json", "tokenizer.model",
    "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json",
}
IDENTITY_FILE = "history_attention.json"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(paths):
    return [
        {"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size, "sha256": sha(p)}
        for p in paths
    ]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def verify_assets(data_root):
    folder = Path(data_root) / DATASET
    records = {key: source_identity([folder / f"Beauty.{key}.json"])[0] for key in ASSETS}
    require({key: row["sha256"] for key, row in records.items()} == ASSETS, "LETTER Beauty asset identity mismatch")
    indices = json.loads((folder / "Beauty.index.json").read_text())
    require(set(indices) == set(map(str, range(12101))), "LETTER Beauty item axis mismatch")
    require(all(len(sid) == 4 for sid in indices.values()), "LETTER Beauty SID length mismatch")
    require(len({tuple(sid) for sid in indices.values()}) == 12088, "LETTER Beauty full SID collision count mismatch")
    return records


def verify_base_model(path):
    path = Path(path).resolve()
    require(path.name == MODEL_REVISION, "LLaMA-7B revision mismatch")
    records = {}
    for name, expected in MODEL_FILES.items():
        row = source_identity([path / name])[0]
        require(row["sha256"] == expected, f"LLaMA-7B hash mismatch for {name}")
        records[name] = row
    historical_path = Path("/data/bwwu/lc-rec/assets/llama-7b") / MODEL_REVISION
    for name, (size, mtime_ns, expected) in MODEL_SHARDS.items():
        member = path / name
        stat = member.stat()
        require(stat.st_size == size, f"LLaMA-7B shard size mismatch for {name}")
        use_historical = path == historical_path and stat.st_mtime_ns == mtime_ns
        if not use_historical:
            require(sha(member) == expected, f"LLaMA-7B shard hash mismatch for {name}")
        records[name] = {
            "path": str(member), "bytes": size, "mtime_ns": stat.st_mtime_ns,
            "sha256": expected,
            "method": "historically verified payload, path/size/mtime rechecked" if use_historical else "current full hash",
        }
    return {"path": str(path), "revision": MODEL_REVISION, "files": records}


def tokenizer_identity(tokenizer, tokens):
    encoded = {token: tokenizer.encode(token, add_special_tokens=False) for token in sorted(tokens)}
    require(len(tokenizer) == VOCAB_SIZE and len(encoded) == NEW_TOKEN_COUNT, "LETTER Beauty tokenizer vocabulary mismatch")
    require(all(len(value) == 1 for value in encoded.values()), "LETTER Beauty SID token split")
    mapping = {token: value[0] for token, value in encoded.items()}
    require(set(mapping.values()) == set(range(32000, VOCAB_SIZE)), "LETTER Beauty SID token range mismatch")
    require(tokenizer.pad_token_id == 0, "LETTER Beauty padding ID mismatch")
    return {
        "vocab_size": VOCAB_SIZE, "new_token_count": NEW_TOKEN_COUNT,
        "token_to_id_sha256": json_digest(mapping),
        "full_vocab_sha256": json_digest(tokenizer.get_vocab()),
        "pad_token_id": tokenizer.pad_token_id,
    }


def verify_execution_profile(profile):
    require(profile == EXECUTION_PROFILE, "LETTER Beauty single-GPU execution profile mismatch")


def write_adapter_identity(adapter, *, mode, scope, run_commit, config_sha256, data_identity,
                           tokenizer_record, base_model_identity, source_identity, checkpoint_source=None):
    root = Path(adapter)
    require(mode in MODES and scope in {"PREFLIGHT", "FORMAL"}, "invalid model mode or checkpoint scope")
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": sha(root / name)} for name in sorted(ADAPTER_MEMBERS)}
    value = {
        "experiment_id": EXPERIMENT_ID, "dataset": DATASET, "mode": mode,
        "checkpoint_purpose": scope.lower(), "model_revision": MODEL_REVISION,
        "run_commit": run_commit, "config_sha256": config_sha256,
        "data_identity": {name: row["sha256"] for name, row in data_identity.items()},
        "tokenizer_identity": tokenizer_record, "base_model_identity": base_model_identity,
        "source_identity": source_identity, "execution_profile": EXECUTION_PROFILE,
        "files": files, "adapter_sha256": files["adapter_model.bin"]["sha256"],
        "checkpoint_source": checkpoint_source,
    }
    atomic_json(root / IDENTITY_FILE, value)
    return value


def verify_adapter_identity(adapter, *, scope, mode=None, run_commit=None):
    root = Path(adapter)
    value = json.loads((root / IDENTITY_FILE).read_text())
    checks = (
        value["experiment_id"] == EXPERIMENT_ID,
        value["dataset"] == DATASET,
        value["mode"] in MODES and (mode is None or value["mode"] == mode),
        value["checkpoint_purpose"] == scope.lower(),
        value["model_revision"] == MODEL_REVISION,
        re.fullmatch(r"[a-f0-9]{40}", value["run_commit"]) is not None,
        run_commit is None or value["run_commit"] == run_commit,
        value["data_identity"] == ASSETS,
        set(value["files"]) == ADAPTER_MEMBERS,
        value["adapter_sha256"] == value["files"]["adapter_model.bin"]["sha256"],
    )
    require(all(checks), "LETTER Beauty history attention adapter metadata mismatch")
    verify_execution_profile(value["execution_profile"])
    for name, row in value["files"].items():
        member = root / name
        require(member.is_file() and not member.is_symlink(), f"adapter member unavailable: {name}")
        require(member.stat().st_size == row["bytes"] and sha(member) == row["sha256"], f"adapter member identity mismatch: {name}")
    return value
