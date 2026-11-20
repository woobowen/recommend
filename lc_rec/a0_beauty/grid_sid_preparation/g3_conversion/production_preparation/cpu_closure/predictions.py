"""Stream the original G3 Beauty predictions and independently verify all 16 metrics."""

from collections import Counter
import csv
import hashlib
import io
from itertools import permutations
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import numpy as np


CONDITIONS = ("normal", "s3", "s4")
USERS = 22363
BEAMS = 20
CUTOFFS = (5, 10, 20)
KEYS = tuple([f"hit@{k}" for k in (1, *CUTOFFS)] +
             [f"{m}@{k}" for m in ("ndcg", "mrr", "ItemHit", "ItemNDCG") for k in CUTOFFS])
PAIR_FIELDS = (
    "row_idx", "user_id", "user_axis_sha256", "target_item_id", "target_sid",
    "target_group_size", "original_interaction_length", "visible_history_length", "history_group",
    "target_training_frequency", "popularity_group", "protocol_version",
)
# The production CSV preserves the row dictionary's insertion order. The
# matrix and summary use canonical KEYS; consume CSV metrics by name.
CSV_METRIC_KEYS = (KEYS[:4] + tuple(f"{m}@{k}" for k in CUTOFFS for m in ("ndcg", "mrr")) +
                   tuple(f"{m}@{k}" for k in CUTOFFS for m in ("ItemHit", "ItemNDCG")))
CSV_FIELDS = PAIR_FIELDS[:-1] + ("first_target_rank", "generation_seconds", PAIR_FIELDS[-1]) + CSV_METRIC_KEYS
AXIS_SHA256 = "7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f"
ASSIGNMENT_SHA256 = "22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088"
TOKEN_MAPPING_SHA256 = "dba960c3166b49d9098c713095273db3b6faa856ae2485ec8eda408ae66e82bc"
OFFICIAL_CCE_SHA256 = "173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d"
METRIC_TOLERANCE = 1e-13
CCE_TOLERANCE = 1e-14
PROTOCOL = "letter_beauty_beam20_v1"
SHUFFLE_PROTOCOL = "GRID-LETTER-SHUFFLE-V1"
DISCOUNT = tuple(1 / math.log2(rank + 1) for rank in range(1, 21))
PERMUTATIONS = {n: tuple(p for p in permutations(range(n)) if p != tuple(range(n))) for n in (3, 4)}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def load_verified(path, expected=None):
    """Read a lightweight artifact once and bind the bytes consumed to its receipt."""
    path = Path(path)
    before = signature(path.stat())
    data = path.read_bytes()
    identity = {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    require(before == signature(path.stat()) and len(data) == before[2], f"Input changed during read: {path}")
    if expected is not None:
        require(all(identity[k] == expected[k] for k in ("bytes", "sha256") if k in expected),
                f"Input identity mismatch: {path}")
        if "path" in expected:
            require(Path(expected["path"]).resolve() == path.resolve(), f"Input path mismatch: {path}")
    return data, identity


def record(path, expected=None):
    content, identity = load_verified(path, expected)
    return json.loads(content), identity


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def score(sids, target, sizes):
    """Reconstruct SID hits/DCG/MRR and collision-expanded item metrics."""
    positions = [ordinal for ordinal, sid in enumerate(sids, 1) if sid == target]
    first = positions[0] if positions else 0
    hits = [float(0 < first <= cutoff) for cutoff in (1, *CUTOFFS)]
    dcg = [math.fsum(DISCOUNT[r - 1] for r in positions if r <= cutoff) for cutoff in CUTOFFS]
    rr = [1 / first if 0 < first <= cutoff else 0.0 for cutoff in CUTOFFS]

    # Each beam before the first match occupies its catalog group's size.
    # An invalid SID still occupies one slot. A repeated prefix is counted
    # again. Only the first occurrence of the target expands to its group.
    group_size = sizes[target]
    prefix_slots = sum(sizes.get(sid, 1) for sid in sids[:first - 1]) if first else 0
    item_counts = [min(group_size, max(0, k - prefix_slots)) if first else 0 for k in CUTOFFS]
    item_hit = [count / group_size for count in item_counts]
    item_dcg = [math.fsum(DISCOUNT[prefix_slots:prefix_slots + count]) / group_size for count in item_counts]
    return hits + dcg + rr + item_hit + item_dcg, first


def expected_permutation(n, row, item):
    # Only [n, row, history ordinal, counter] varies. Integer arithmetic is
    # exact so the frozen rejection sampling remains unchanged.
    options = PERMUTATIONS[n]
    ceiling = (1 << 256) // len(options) * len(options)
    counter = 0
    while True:
        seed_material = f'["{SHUFFLE_PROTOCOL}",42,{n},{row},{item},{counter}]'.encode()
        draw = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
        if draw < ceiling:
            code = draw % len(options)
            return list(options[code] + ((3,) if n == 3 else ())), code, counter
        counter += 1


def verify_input(row, history, item_tokens, n, original_fingerprint):
    idx = row["row_idx"]
    original, actual, mask = (row[key] for key in ("original_input_token_ids", "input_token_ids", "attention_mask"))
    require(len(actual) == len(original) == len(mask) and len(mask) <= 2048 and set(mask) <= {0, 1},
            "Invalid sequence/attention mask")
    require(row["input_nonpadding_tokens"] == sum(mask) and row["preflight_extra_left_padding"] == 0,
            "Padding/active-token evidence")
    active = [i for i, enabled in enumerate(mask) if enabled]
    require(active and len(active) == active[-1] - active[0] + 1, "Noncontiguous history input mask")
    require(all(type(token) is int and 0 <= token < 32786 for token in original), "Invalid input token ID")
    fingerprint = digest([original, mask])
    if original_fingerprint is not None:
        require(fingerprint == original_fingerprint, "Original tokens/attention mask differ from Normal")
    locations = [i for i, token in enumerate(original) if token >= 32000]
    require(len(locations) == 4 * len(history) and all(mask[i] for i in locations),
            "Input has missing/extra history SID symbols or SID padding")
    computed = original.copy()
    groups = []
    for item, key in enumerate(history):
        positions = locations[4 * item:4 * item + 4]
        expected = item_tokens[key]
        require(positions == list(range(positions[0], positions[0] + 4)) and
                [original[i] for i in positions] == expected, f"History item or position differs at ordinal {item}")
        if n:
            perm, code, counter = expected_permutation(n, idx, item)
            shuffled = [expected[j] for j in perm]
            for i, token in zip(positions, shuffled):
                computed[i] = token
            groups.append({
                "visible_history_item_index": item, "start": positions[0], "positions": positions,
                "permutation": perm, "permutation_id": code, "rejection_counter": counter,
                "original_ids": expected, "shuffled_ids": shuffled, "nonidentity": True,
            })
    require(actual == computed, "Actual input differs from deterministic within-item permutation")
    if n:
        expected_evidence = {
            "protocol_version": SHUFFLE_PROTOCOL, "seed": 42, "shuffle_n": n, "row_idx": idx,
            "history_group_count": len(history), "input_changed": original != computed,
            "nonhistory_unchanged": True, "attention_mask_unchanged": True, "sid_multiset_conserved": True,
            "original_sha256": digest(original), "shuffled_sha256": digest(computed),
            "attention_mask_sha256": digest(mask), "groups": groups,
        }
        require(row["shuffle"] == {
            "n": n, "seed": 42, "protocol": SHUFFLE_PROTOCOL, "row_idx": idx,
            "original_input_token_ids": original, "attention_mask": mask, "evidence": expected_evidence,
        }, "Saved Shuffle group protocol/evidence differs from replay")
    else:
        require("shuffle" not in row, "Normal row declares an intervention")
    prefill = row["actual_model_prefill"]
    positions, current = [], -1
    for visible in mask:
        current += visible
        positions.append(current if visible else 1)
    require(prefill == {"input_ids": actual, "attention_mask": mask, "position_ids": positions,
                       "initial_cache_length": 0, "beam_expansion": BEAMS},
            "Saved prefill differs in tokens, mask, positions, cache, or beam expansion")
    require(row["lookup_equivalence_exact"] is True and row["original_callback_verified_all_tokens"] is True,
            "Missing native generation lookup/callback evidence")
    return fingerprint, len(history), bool(groups and original != computed)


def reconstruct_candidates(row, symbol_by_id, allowed_ids):
    candidates = row["candidates"]
    require(len(candidates) == BEAMS, "Beam20 candidate count")
    decoded = []
    for order, candidate in enumerate(candidates):
        raw = candidate["raw_decoded"]
        sid = raw.rsplit("Response:", 1)[-1].strip().translate({32: None})
        value = candidate["sequence_score"]
        require(type(value) in (int, float) and math.isfinite(value) and
                candidate["generation_order"] == order and candidate["sid"] == sid,
                f"Candidate {order} order/SID/score differs")
        suffix = candidate["generated_token_ids"]
        require(candidate["eos_position"] == 4 and len(suffix) >= 5 and suffix[4] == 2 and
                set(suffix[5:]) <= {0, 2}, f"Candidate {order} EOS/suffix differs")
        require(all(token in allowed_ids[i] for i, token in enumerate(suffix[:4])) and
                "".join(symbol_by_id[token] for token in suffix[:4]) == sid,
                f"Candidate {order} token callback or decoded SID differs")
        decoded.append(sid)
    order = sorted(range(BEAMS), key=lambda i: -candidates[i]["sequence_score"])
    ranked = tuple(decoded[i] for i in order)
    relevance = [int(sid == row["target"]) for sid in ranked]
    require(order == row["sorted_generation_order"] and relevance == row["relevance"],
            "Saved stable order or relevance differs")
    return ranked


def input_metadata(plan, asset_paths):
    data = {}
    identities = {}
    for name, checksum in plan["asset_sha256"].items():
        path = (Path(asset_paths[name]) if asset_paths is not None else
                Path(plan["data_root"]) / "Beauty" / f"Beauty.{name}.json")
        data[name], identities[name] = record(path, {"sha256": checksum})
    index, inter = data["index"], data["inter"]
    require(len(index) == 12101 and len(inter) == USERS and set(data["item"]) == set(index), "Fixed item/user axes")
    require(all(len(sid) == 4 for sid in index.values()), "Four-token G3 index")
    sid_by_item = {item: "".join(tokens) for item, tokens in index.items()}
    sizes = Counter(sid_by_item.values())
    require(len(sizes) == 12101, "Frozen G3 full SID collision invariant")
    users = list(inter)
    require(digest(users) == AXIS_SHA256, "Frozen user-axis digest")
    require(all(seq and all(str(item) in index for item in seq) for seq in inter.values()), "Unknown item in interactions")
    frequencies = Counter(str(item) for seq in inter.values() for item in seq[:-2])
    levels = sorted(frequencies.values())
    q1, q2 = levels[math.ceil(len(levels) / 3) - 1], levels[math.ceil(2 * len(levels) / 3) - 1]
    symbol_to_id = {token: i for i, token in enumerate(sorted({t for sid in index.values() for t in sid}), 32000)}
    require(len(symbol_to_id) == 786 and digest(symbol_to_id) == TOKEN_MAPPING_SHA256, "Fixed SID token-ID mapping")
    item_tokens = {item: [symbol_to_id[token] for token in parts] for item, parts in index.items()}
    allowed_ids = tuple({tokens[i] for tokens in item_tokens.values()} for i in range(4))
    require(tuple(map(len, allowed_ids)) == (256, 256, 256, 18), "Frozen positional callback catalog")
    return {
        "identities": identities, "users": users, "interactions": inter, "sids": sid_by_item, "sizes": sizes,
        "frequencies": frequencies, "tertiles": (q1, q2), "symbol_to_id": symbol_to_id,
        "symbol_by_id": {i: token for token, i in symbol_to_id.items()},
        "item_tokens": item_tokens, "allowed_ids": allowed_ids,
    }


def rank_metadata(plan, condition, run_commit, adapter_files, meta, rank):
    require(rank["RUN_COMMIT"] == run_commit and rank["condition"] == condition and rank["scope"] == "FORMAL" and
            rank["status"] == "PASS" and rank["rank"] == 0 and rank["world_size"] == 1 and
            rank["execution_topology"] == "SINGLE_GPU" and rank["execution_profile"] == plan["execution_profile"] and
            rank["experiment_id"] == plan["experiment_id"] and rank["preflight_only"] is False,
            "Formal rank model/condition/source identity")
    require(rank["adapter_files"] == adapter_files and rank["model_identity_sha256"] == adapter_files["g3_identity.json"]["sha256"]
            and {k: v["sha256"] for k, v in rank["data_identity"].items()} == plan["asset_sha256"],
            "Rank adapter/data identity")
    require(rank["tokenizer"]["added_tokens"] == meta["symbol_to_id"] and
            rank["tokenizer_identity"]["token_to_id_sha256"] == TOKEN_MAPPING_SHA256 and
            rank["tokenizer_identity"]["vocab_size"] == 32786, "Rank tokenizer identity")
    generation = rank["generation"]
    expected = {"num_beams": 20, "num_return_sequences": 20, "max_new_tokens": 10,
                "early_stopping": True, "prompt_id": 0, "filter_items": False, "per_rank_batch_size": 1,
                "oom_beam_reduction": False, "prefix": "fixed original per-position callback; no catalog Trie"}
    require(all(generation[k] == v for k, v in expected.items()), "Rank native Beam20 generation policy")
    assignment = rank["assigned_row_indices"]
    require(rank["assigned_users"] == rank["completed_users"] == len(assignment) == USERS and
            set(assignment) == set(range(USERS)) and digest(assignment) == rank["assignment_sha256"] == ASSIGNMENT_SHA256,
            "Rank fixed sampler order/coverage identity")
    n = {"normal": 0, "s3": 3, "s4": 4}[condition]
    require(("--verify-normal-off" in rank["argv"]) == (n == 0), "Normal/OFF argv")
    checks = rank["normal_off_same_process_exact"]
    if n:
        require(rank["shuffle"] == {"n": n, "seed": 42, "protocol": SHUFFLE_PROTOCOL} and not checks,
                "Rank Shuffle condition protocol")
    else:
        require("shuffle" not in rank and len(checks) == USERS and
                all(r["row_idx"] == idx and r["same_process_exact"] is True for r, idx in zip(checks, assignment)),
                "Normal/OFF full direct exact comparison/assignment evidence")
    return assignment, n


def identity_fields(meta, idx):
    user = meta["users"][idx]
    seq = meta["interactions"][user]
    item = str(seq[-1])
    target = meta["sids"][item]
    history = [str(value) for value in seq[:-1][-20:]]
    size = len(history)
    hgroup = "0-5" if size <= 5 else "6-10" if size <= 10 else "11-20"
    count = meta["frequencies"][item]
    q1, q2 = meta["tertiles"]
    pgroup = "UNSEEN" if count == 0 else "LONG_TAIL" if count <= q1 else "MID" if count <= q2 else "POPULAR"
    fields = (str(idx), user, AXIS_SHA256, item, target, str(meta["sizes"][target]),
              str(len(seq)), str(size), hgroup, str(count), pgroup, PROTOCOL)
    return fields, history


def evaluate_shard(path, sealed, rank, assignment, n, meta, official, original_inputs, samples=None):
    """Single streaming pass over the prediction payload in sampler order."""
    before = signature(path.stat())
    require(Path(sealed["path"]).resolve() == path.resolve() and before[2] == sealed["bytes"], "Sealed prediction path/size")
    payload_sha = hashlib.sha256()
    tally = {k: Counter() for k in CUTOFFS}
    coverage = {k: set() for k in CUTOFFS}
    first_bins = Counter()
    matrix = np.empty((USERS, len(KEYS)), dtype=np.float64)
    ranked = [None] * USERS
    fields = [None] * USERS
    signatures = [None] * USERS
    first_ranks = np.empty(USERS, dtype=np.int64)
    generation_seconds = np.empty(USERS, dtype=np.float64)
    max_cce = 0.0
    worst_cce = None
    cce_nonzero = 0
    group_count = changed_users = seen = 0
    identifiers = {"condition": rank["condition"], "scope": "FORMAL", "rank": 0,
                   "world_size": 1, "prompt_id": 0, "execution_topology": "SINGLE_GPU",
                   "execution_profile": rank["execution_profile"], "model_identity_sha256": rank["model_identity_sha256"]}
    sizes = meta["sizes"]
    with path.open("rb", buffering=1024 ** 2) as stream:
        require(signature(os.fstat(stream.fileno())) == before, "Prediction changed before reading")
        for seen, raw in enumerate(stream, 1):
            payload_sha.update(raw)
            idx = None
            try:
                row = json.loads(raw)
                idx = row["row_idx"]
                require(seen <= USERS and type(idx) is int and idx == assignment[seen - 1] and ranked[idx] is None,
                        "Duplicate, out-of-range, or misordered user row")
                require(all(row[k] == value for k, value in identifiers.items()), "Prediction scope/model/condition")
                row_fields, history = identity_fields(meta, idx)
                require((row["user_id"], row["target_item_id"], row["target"]) ==
                        (row_fields[1], row_fields[3], row_fields[4]), "Prediction user or target does not match fixed source")
                signatures[idx], groups, changed = verify_input(
                    row, history, meta["item_tokens"], n, None if original_inputs is None else original_inputs[idx])
                group_count += groups
                changed_users += changed
                sids = reconstruct_candidates(row, meta["symbol_by_id"], meta["allowed_ids"])
                value, first = score(sids, row["target"], sizes)
                expected = official(sids, row["target"], sizes, CUTOFFS)
                for j, key in enumerate(KEYS[-6:]):
                    diff = abs(value[-6 + j] - expected[key])
                    cce_nonzero += diff != 0
                    require(diff <= CCE_TOLERANCE, f"Official CCE differs for {key}: {diff}")
                    if diff > max_cce:
                        max_cce, worst_cce = diff, {"row_idx": idx, "metric": key}
                duration = row["generation_seconds"]
                require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                        "Prediction generation timing")
                matrix[idx] = value
                ranked[idx] = tuple(sys.intern(sid) for sid in sids)
                fields[idx], first_ranks[idx], generation_seconds[idx] = row_fields, first, duration
                first_bins[str(first) if first else "miss"] += 1
                for k in CUTOFFS:
                    candidates = sids[:k]
                    valid = [sid for sid in candidates if sid in sizes]
                    dup = k - len(set(candidates))
                    colliding = sum(sizes[sid] > 1 for sid in valid)
                    tally[k].update(candidates=k, valid=len(valid), invalid=k - len(valid),
                                    duplicates=dup, colliding=colliding, users_invalid=int(len(valid) < k),
                                    users_duplicate=int(dup > 0), users_colliding=int(colliding > 0))
                    coverage[k].update(valid)
                if samples is not None and seen <= 8:
                    samples.write(raw)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: line {seen}, row_idx {idx}: {exc}") from exc
        require(signature(os.fstat(stream.fileno())) == before, "Prediction file changed during read")
    actual_digest = payload_sha.hexdigest()
    require(signature(path.stat()) == before, f"Prediction path changed during streaming: {path}")
    require(actual_digest == sealed["sha256"],
            f"Prediction content identity mismatch for {path}: {actual_digest}, expected {sealed['sha256']}")
    require(seen == USERS and all(r is not None for r in ranked), f"User coverage incomplete: {seen}/{USERS}")
    sid_stats = {}
    for k, count in tally.items():
        unique = len(coverage[k])
        sid_stats[str(k)] = {
            **count, "valid_rate": count["valid"] / (USERS * k), "invalid_rate": count["invalid"] / (USERS * k),
            "duplicate_rate": count["duplicates"] / (USERS * k), "collision_candidate_rate": count["colliding"] / (USERS * k),
            "users_invalid_rate": count["users_invalid"] / USERS, "users_duplicate_rate": count["users_duplicate"] / USERS,
            "users_colliding_rate": count["users_colliding"] / USERS, "unique_valid_sids": unique,
            "coverage_denominator": len(sizes), "unique_valid_sid_coverage": unique / len(sizes),
        }
    return {
        "matrix": matrix, "ranked": ranked, "per_user_fields": fields, "input_signatures": signatures,
        "first_ranks": first_ranks, "generation_seconds": generation_seconds,
        "prediction_signature": before, "sid_statistics": sid_stats,
        "histogram": {str(i): first_bins[str(i)] for i in range(1, 21)} | {"miss": first_bins["miss"]},
        "max_cce": max_cce, "cce_worst": worst_cce, "cce_nonzero": cce_nonzero,
        "history_group_count": group_count, "input_changed_users": changed_users,
    }


def difference_report(actual, archived, keys):
    diff = np.abs(actual - archived)
    locations = {}
    for j, key in enumerate(keys):
        row = int(diff[:, j].argmax())
        maximum = float(diff[row, j])
        locations[key] = {
            "max_absolute_difference": maximum,
            "max_difference_row_idx": row if maximum else None,
            "actual_at_max": float(actual[row, j]) if maximum else None,
            "archived_at_max": float(archived[row, j]) if maximum else None,
            "nonzero_rows": int(np.count_nonzero(diff[:, j])),
            "locations_sample": np.flatnonzero(diff[:, j])[:10].tolist(),
        }
    return locations


def compare_result(primary, result, production, output):
    require(production["users"] == USERS and production["row_indices"] == list(range(USERS)) and
            tuple(production["metrics"]) == KEYS and production["user_axis_sha256"] == AXIS_SHA256 and
            production["official_cce_full_user_comparisons"] == USERS and
            production["protocol_version"] == PROTOCOL, "Original production evaluation protocol missing")
    paths = {Path(v["path"]).name: v for v in production["per_user_files"]}
    require(set(paths) == {"per_user.csv", "per_user_metrics.npz"}, "Production per-user receipt")
    arrays, npz_id = load_verified(primary / "per_user_metrics.npz", paths["per_user_metrics.npz"])
    csv_bytes, csv_id = load_verified(primary / "per_user.csv", paths["per_user.csv"])
    files = {"per_user.csv": csv_id, "per_user_metrics.npz": npz_id}
    with np.load(io.BytesIO(arrays), allow_pickle=False) as data:
        original = data["values"]
        expected = ("keys", "values", "row_idx", "user_axis_sha256")
        require(set(data.files) == set(expected) and tuple(data["keys"].tolist()) == KEYS and
                np.array_equal(data["row_idx"], np.arange(USERS, dtype=np.int64)) and
                data["row_idx"].dtype == np.int64 and str(data["user_axis_sha256"]) == AXIS_SHA256 and
                original.dtype == np.float64 and original.shape == (USERS, len(KEYS)) and
                np.isfinite(original).all(), "Production NPZ user axis, dtype, shape, or metric order differs")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode(), newline=""))
    require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "Production CSV identity fields or metric order")
    matrix_csv = np.empty_like(original)
    count = 0
    for count, entry in enumerate(reader, 1):
        row = count - 1
        require(count <= USERS and tuple(entry[k] for k in PAIR_FIELDS) == result["per_user_fields"][row],
                f"Original CSV row {row} differs in fixed user/target/group/protocol fields")
        require(entry["first_target_rank"] == str(result["first_ranks"][row]) and
                float(entry["generation_seconds"]) == result["generation_seconds"][row],
                f"Original CSV row {row} differs in first target rank or generation time")
        matrix_csv[row] = [float(entry[k]) for k in KEYS]
    require(count == USERS and np.isfinite(matrix_csv).all(), "Production CSV user coverage/finite")
    exact = difference_report(matrix_csv, original, KEYS)
    comparison = difference_report(result["matrix"], original, KEYS)
    summary = {key: math.fsum(result["matrix"][:, j]) / USERS for j, key in enumerate(KEYS)}
    mean_deltas = {key: summary[key] - production["metrics"][key] for key in KEYS}
    checks = {
        "csv_vs_npz_by_metric": exact, "independent_vs_npz_by_metric": comparison,
        "independent_vs_production_mean_delta": mean_deltas,
        "sid_diagnostics_exact": result["sid_statistics"] == production["sid_statistics"],
        "histogram_exact": result["histogram"] == production["first_target_rank_users"],
        "metric_mean_tolerance_strict": METRIC_TOLERANCE,
        "cce_per_user_tolerance_inclusive": CCE_TOLERANCE,
    }
    failed = (
        any(v["nonzero_rows"] for v in exact.values()) or
        any(comparison[k]["nonzero_rows"] for k in KEYS[:4]) or
        any(comparison[k]["max_absolute_difference"] > CCE_TOLERANCE for k in KEYS[-6:]) or
        any(comparison[k]["max_absolute_difference"] >= METRIC_TOLERANCE for k in KEYS[4:-6]) or
        any(abs(d) >= METRIC_TOLERANCE for d in mean_deltas.values()) or
        not checks["sid_diagnostics_exact"] or not checks["histogram_exact"]
    )
    if failed:
        write_json(output.with_name(output.stem + "-mismatch.json"), checks)
        raise ValueError(f"Original and reconstructed metrics/diagnostics disagree: {primary}")
    return original, files, summary, checks


def receipt_member(receipt, path):
    matches = [row for row in receipt["files"] if Path(row["path"]).resolve() == path.resolve()]
    require(len(matches) == 1, f"Original stage result missing a unique identity: {path}")
    return matches[0]


def verify_stage(result, output, stage, run_commit):
    require(result["verified"] is True and result["scope"] == "FORMAL" and
            result["run_commit"] == run_commit and result["stage"] == stage and
            Path(result["output"]).resolve() == output.resolve(),
            f"Original stage result identity: {stage}")


def audit_all(plan, stage_outputs, output_dir, *, run_commit, adapter_files, asset_paths=None,
              prediction_pins=None, save_samples=False):
    """Return verified matrices/ordered recommendations for CPU pairing and persist audit evidence."""
    output = Path(output_dir) / "independent"
    source = Path(plan["release"])
    require(output.parent.resolve().is_relative_to(Path(plan["root"]).resolve()) is False,
            "Independent results cannot be placed in the original formal root")
    output.mkdir(exist_ok=False)
    common_started = time.monotonic()
    meta = input_metadata(plan, asset_paths)
    cce_path = source / "lc_rec/letter/review/final_audit_20261113/external/official_cce.py"
    cce_code, cce_identity = load_verified(cce_path, {"sha256": OFFICIAL_CCE_SHA256})
    # Only the pinned reference CCE comparator is loaded. The independent
    # evaluator owns parsing, ordering, formulas, and aggregation.
    cce = types.ModuleType("original_cce_reference")
    exec(compile(cce_code, str(cce_path), "exec"), cce.__dict__)
    preparation_seconds = time.monotonic() - common_started
    results = {}
    original_inputs = None
    common_binding = None
    for c in CONDITIONS:
        started = time.monotonic()
        generation_output = Path(stage_outputs[c])
        evaluation_output = Path(stage_outputs[c + "_eval"])
        test, primary = generation_output / "test", evaluation_output / "primary"
        gen, gen_id = record(generation_output / "stage_result.json")
        ev, ev_id = record(evaluation_output / "stage_result.json")
        verify_stage(gen, generation_output, c, run_commit)
        verify_stage(ev, evaluation_output, c + "_eval", run_commit)
        rank, rank_id = record(test / "rank0.json", receipt_member(gen, test / "rank0.json"))
        assignment, n = rank_metadata(plan, c, run_commit, adapter_files, meta, rank)
        current_binding = {k: rank[k] for k in ("tokenizer_identity", "generation", "allowed_callback_sha256")}
        if common_binding is None:
            common_binding = current_binding
        require(common_binding == current_binding, "Frozen tokenizer, generation arguments or callback differ by condition")
        official, production_id = record(primary / "metrics.json", ev["proof"]["metrics"])
        require(official["status"] == "PASS" and official["scope"] == "FORMAL" and
                official["run_commit"] == official["evaluation_commit"] == run_commit and
                official["world_size"] == 1 and official["execution_profile"] == plan["execution_profile"] and
                {k: v["sha256"] for k, v in official["assets"].items()} == plan["asset_sha256"] and
                official["official_cce"]["sha256"] == cce_identity["sha256"],
                "Production offline configured scope/model/data/reference identity")
        cfg_path = evaluation_output / "evaluation_config.json"
        config, config_id = record(cfg_path, official["config"])
        require(config["original_run_commit"] == run_commit and config["asset_sha256"] == plan["asset_sha256"] and
                config["user_axis_sha256"] == AXIS_SHA256 and config["beams"] == BEAMS and
                config["filter_items"] is False and config["deduplicate"] is False,
                "Production offline actual config identity")
        proof = gen["proof"]["full_user_audit"]
        require(proof["status"] == "PASS" and proof["actual_test_users"] == USERS and
                all(len(items) == 1 for items in (gen["proof"]["prediction_files"], proof["prediction_files"],
                    ev["proof"]["prediction_proof"]["prediction_files"], official["prediction_files"])) and
                gen["proof"]["adapter_files"] == adapter_files, "Original stage proof source and scope")
        sealed = gen["proof"]["prediction_files"][0]
        for other in (proof["prediction_files"][0], ev["proof"]["prediction_proof"]["prediction_files"][0],
                      official["prediction_files"][0]):
            require(all(sealed[k] == other[k] for k in ("path", "bytes", "sha256")), "Prediction receipt mismatch")
        if prediction_pins is not None:
            require(all(sealed[k] == prediction_pins[c][k] for k in ("bytes", "sha256")), "Prediction fixed task pin mismatch")
        pred_path = test / "predictions.rank0.jsonl"
        sample_path = output / f"{c}-samples.jsonl"
        stream_start = time.monotonic()
        print(json.dumps({"stage": "independent_full_prediction_recomputation", "condition": c,
                          "event": "started", "bytes": sealed["bytes"]}), flush=True)
        if save_samples:
            with sample_path.open("xb") as samples:
                result = evaluate_shard(pred_path, sealed, rank, assignment, n, meta, cce.compute_cce_metrics, original_inputs, samples)
        else:
            result = evaluate_shard(pred_path, sealed, rank, assignment, n, meta, cce.compute_cce_metrics, original_inputs)
        compare_start = time.monotonic()
        if original_inputs is None:
            original_inputs = result["input_signatures"]
        rank_artifact = {"identity": rank_id, "assigned_users": USERS, "assignment_sha256": ASSIGNMENT_SHA256,
                         "scope": "FORMAL", "condition": c, "prefill_rows_verified": USERS,
                         "shuffle_n": n, "history_groups_checked": result["history_group_count"],
                         "input_changed_users": result["input_changed_users"],
                         "normal_off_exact_records": len(rank["normal_off_same_process_exact"])}
        if c == "normal":
            rank_artifact["complete_off_check_sha256"] = digest(rank["normal_off_same_process_exact"])
        output_report = output / f"{c}.json"
        original, identities, means, comparison = compare_result(primary, result, official, output_report)
        top = result["sid_statistics"]["20"]
        gen_diag = proof["anomalies"]
        require(top["candidates"] == gen_diag["candidate_count"] and
                top["invalid"] == gen_diag["invalid_complete_sid_candidates"] and
                top["duplicates"] == gen_diag["duplicate_candidate_occurrences"] and
                top["colliding"] == gen_diag["colliding_sid_candidate_occurrences"] and
                top["users_invalid"] == gen_diag["users_with_invalid_sid"] and
                top["users_duplicate"] == gen_diag["users_with_duplicate_sid"] and
                top["users_colliding"] == gen_diag["users_with_colliding_sid_candidate"] and
                all(abs(means[key] - val) < METRIC_TOLERANCE for key, val in proof["metrics"].items()),
                "Original generation receipt anomaly or metric comparison")
        require(official["collision_catalog"]["items"] == official["collision_catalog"]["unique_sids"] == len(meta["sizes"]) and
                official["collision_catalog"]["groups"] == official["collision_catalog"]["target_collision_users"] == 0 and
                official["collision_catalog"]["maximum_group_size"] == 1 and
                gen_diag["targets_with_colliding_sid"] == 0, "Catalog and target collision diagnostics")
        for k in (1, *CUTOFFS):
            target_hits = sum(result["histogram"][str(i)] for i in range(1, k + 1))
            require(target_hits == round(means[f"hit@{k}"] * USERS), f"Hit@{k} histogram marginal")
        np.savez_compressed(output / f"{c}.npz", values=result["matrix"], keys=np.asarray(KEYS),
                            row_idx=np.arange(USERS, dtype=np.int64), user_axis_sha256=np.asarray(AXIS_SHA256),
                            first_target_rank=result["first_ranks"])
        audit = {
            "status": "PASS", "run_commit": run_commit, "condition": c, "users": USERS, "keys": KEYS,
            "user_axis_sha256": AXIS_SHA256, "candidates_per_user": BEAMS, "total_candidates": BEAMS * USERS,
            "metrics": means, "adapter_files": adapter_files, "assets": meta["identities"], "rank": rank_artifact,
            "source_receipts": {"generation": gen_id, "evaluation": ev_id, "metrics": production_id,
                                "evaluation_config": config_id, "per_user": identities, "official_cce": cce_identity},
            "prediction": {**sealed, "content_rehashed_while_streaming": True, "signature": result["prediction_signature"]},
            "sid_statistics": result["sid_statistics"], "first_target_rank_users": result["histogram"],
            "collision_catalog": official["collision_catalog"], "comparison": comparison,
            "official_cce_per_user_comparison": {"max_absolute_difference": result["max_cce"],
                "max_difference_location": result["cce_worst"], "nonzero_values": result["cce_nonzero"],
                "users": USERS, "tolerance_inclusive": CCE_TOLERANCE},
            "max_abs_vs_original_per_user": max(v["max_absolute_difference"] for v in comparison["independent_vs_npz_by_metric"].values()),
            "max_abs_vs_original_aggregate": max(map(abs, comparison["independent_vs_production_mean_delta"].values())),
            "elapsed_seconds": time.monotonic() - started,
            "phase_seconds": {
                "shared_data_token_mapping_and_reference_cce_identity": preparation_seconds,
                "condition_receipts_rank_and_config_binding": stream_start - started,
                "full_prediction_stream_hash_replay_ranking_and_scoring": compare_start - stream_start,
                "csv_npz_metrics_diagnostics_comparison_and_output": time.monotonic() - compare_start,
            },
        }
        write_json(output_report, audit)
        results[c] = {"matrix": result["matrix"], "original_matrix": original, "ranked": result["ranked"],
                      "metrics": means, "per_user_fields": result["per_user_fields"], "audit": audit,
                      "paired_input_identities": {str(primary): {name: info["sha256"] for name, info in identities.items()}}}
        print(json.dumps({"stage": "independent_full_prediction_recomputation", "condition": c,
                          "event": "complete", "elapsed_seconds": audit["elapsed_seconds"]}), flush=True)
    return results
