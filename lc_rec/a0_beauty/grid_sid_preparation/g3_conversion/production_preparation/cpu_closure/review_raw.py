#!/usr/bin/env python3
"""Non-author readback from original inputs, separately from closure producers."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import hashlib
from itertools import accumulate, permutations, zip_longest
import json
import math
from pathlib import Path
import time

import numpy as np


USERS = 22363
BEAMS = 20
CONDITIONS = ("normal", "s3", "s4")
HIT_K = (1, 5, 10, 20)
K_VALUES = (5, 10, 20)
KEYS = [f"hit@{k}" for k in HIT_K] + [
    f"{name}@{k}" for name in ("ndcg", "mrr", "ItemHit", "ItemNDCG") for k in K_VALUES
]
PAIR_FIELDS = ("row_idx", "user_id", "user_axis_sha256", "target_item_id", "target_sid",
               "target_group_size", "original_interaction_length", "visible_history_length",
               "history_group", "target_training_frequency", "popularity_group", "protocol_version")
PLAN_PIN = "a56076cf07d67bc088442dc35f863134081b455f551fc45d2c6bfb2bfd246407"
CANON_PIN = "526987fa87f0feb565a7a9f42bc58f42e3121b9d9a13cea4613e47f842c394b6"
RUN_PIN = "21d495b0be5b606853ec7cec68a3eaa1b58acbd7"
AXIS_PIN = "7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f"
MODEL_PIN = "3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915"
WEIGHT_PIN = "d3a179b59ba7a7218d6e1767884fd1c5961fffb9b756ea5f8bc5522775db6a28"
TOKENIZER_MAP_PIN = "dba960c3166b49d9098c713095273db3b6faa856ae2485ec8eda408ae66e82bc"
PREDICTIONS = {
    "normal": (400029854, "a2247ec7016c4d88c08957fea6c0338515a124ca316245e9592ac260e8df730a"),
    "s3": (476894106, "78f26e6336ffe16b6fc0f3ed531f2c5b1d0dd2a6ef397e5fbbd1061593265ae2"),
    "s4": (476966283, "5faf3470974ac6906765c978475a2379111f3488fa7691ceacea38a089207fa3"),
}
SHUFFLE_PROTOCOL = "GRID-LETTER-SHUFFLE-V1"
GROUP_PERMUTATIONS = {n: list(permutations(range(n)))[1:] for n in (3, 4)}
DISCOUNTS = tuple(1 / math.log2(i + 1) for i in range(1, BEAMS + 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_digest(value, *, sort=False):
    return digest(json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=sort).encode())


def read_json(path):
    return json.loads(path.read_bytes())


def pair_delta(left, right):
    return max(abs(left - right).ravel(), default=0.0)


def payload_identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def stage_output(state, root, stage):
    record = state["stages"][stage]
    attempt = state["attempts"][record["attempt_index"]]
    out = Path(record["output"]).resolve(strict=True)
    require(out.is_relative_to(root / "attempts" / stage), f"{stage} output scope")
    require(attempt["stage"] == stage and attempt["attempt"] == 0, f"{stage} attempt identity")
    return out, attempt


def metric_vector(ordered, target, sizes):
    target_positions = [i for i, sid in enumerate(ordered, 1) if sid == target]
    first = target_positions[0] if target_positions else BEAMS + 1
    catalog_ranks = list(accumulate(sizes.get(sid, 1) for sid in ordered))
    item_prefix = catalog_ranks[first - 2] if 1 < first <= BEAMS else 0
    target_size = sizes[target]
    item_slots = range(item_prefix + 1, item_prefix + 1 + target_size) if first <= BEAMS else ()
    ndcg = [math.fsum(DISCOUNTS[i - 1] for i in target_positions if i <= k) for k in K_VALUES]
    item_hit = [sum(i <= k for i in item_slots) / target_size for k in K_VALUES]
    item_ndcg = [math.fsum(DISCOUNTS[i - 1] for i in item_slots if i <= k) / target_size for k in K_VALUES]
    metric = (
        [int(first <= k) for k in HIT_K] + ndcg + [1 / first if first <= k else 0 for k in K_VALUES]
        + item_hit + item_ndcg
    )
    return metric, first if target_positions else 0


def deterministic_shuffle(n, user_index, item_index):
    choices = GROUP_PERMUTATIONS[n]
    bound = (1 << 256) - (1 << 256) % len(choices)
    counter = 0
    while True:
        value = int(json_digest([SHUFFLE_PROTOCOL, 42, n, user_index, item_index, counter]), 16)
        if value < bound:
            selected = choices[value % len(choices)]
            return selected + ((3,) if n == 3 else ()), value % len(choices), counter
        counter += 1


def verify_model_input(name, row, normal, history, token_map):
    original = row["original_input_token_ids"]
    mask = row["attention_mask"]
    actual = row["input_token_ids"]
    prefill = row["actual_model_prefill"]
    i = row["row_idx"]
    require(len(original) == len(mask) == len(actual) and all(x in (0, 1) for x in mask), (name, i, "mask"))
    require((original, mask) == (normal["input_token_ids"], normal["attention_mask"]), (name, i, "base input"))
    require(prefill["input_ids"] == actual and prefill["attention_mask"] == mask
            and prefill["beam_expansion"] == 20 and prefill["initial_cache_length"] == 0
            and prefill["position_ids"] == [p - 1 if active else 1 for p, active in zip(accumulate(mask), mask)],
            (name, i, "recorded model prefill"))
    require(row["lookup_equivalence_exact"] is True and row["original_callback_verified_all_tokens"] is True,
            (name, i, "recorded model lookup and positional callback"))
    sid_ids = [(position, token) for position, token in enumerate(original) if token >= 32000]
    flat = [token_map[symbol] for item in history for symbol in item]
    require([token for _, token in sid_ids] == flat, (name, i, "fixed dataset history in native prompt"))
    if name == "normal":
        require("shuffle" not in row, (i, "Normal bypass"))
        return len(history), 0
    n = int(name[-1])
    certificate = row["shuffle"]
    require((certificate["n"], certificate["protocol"], certificate["seed"], certificate["row_idx"]) ==
            (n, SHUFFLE_PROTOCOL, 42, i), (name, i, "seed/scope"))
    require((certificate["original_input_token_ids"], certificate["attention_mask"]) == (original, mask),
            (name, i, "shuffle input and mask"))
    groups = certificate["evidence"]["groups"]
    require(len(groups) == len(history), (name, i, "history group count"))
    expected = original.copy()
    swaps = 0
    for j, group in enumerate(groups):
        tokens = sid_ids[4*j:4*j+4]
        positions = [pos for pos, _ in tokens]
        old = [t for _, t in tokens]
        perm, index, counter = deterministic_shuffle(n, i, j)
        new = [old[k] for k in perm]
        require(group["visible_history_item_index"] == j and positions == list(range(positions[0], positions[0]+4)) and
                (group["positions"], group["original_ids"], group["shuffled_ids"]) == (positions, old, new) and
                (group["permutation"], group["permutation_id"], group["rejection_counter"]) == (list(perm), index, counter),
                (name, i, j, "deterministic shuffle evidence"))
        expected[positions[0]:positions[-1]+1] = new
        swaps += old[-1] != new[-1]
    require(actual == expected, (name, i, "fixed non-history and model input"))
    return len(history), swaps


def review_ranks(test_dirs, users, plan, completion):
    ranks = {}
    hashes = {}
    for name, folder in zip(CONDITIONS, test_dirs, strict=True):
        path = folder / "rank0.json"
        raw = path.read_bytes()
        rank = json.loads(raw)
        require((rank["status"], rank["scope"], rank["condition"], rank["RUN_COMMIT"]) ==
                ("PASS", "FORMAL", name, RUN_PIN), (name, "rank identity"))
        require(rank["model_identity_sha256"] == MODEL_PIN
                and rank["adapter_files"]["adapter_model.bin"]["sha256"] == WEIGHT_PIN
                and rank["adapter_files"]["g3_identity.json"]["sha256"] == MODEL_PIN,
                (name, "same trained model"))
        require(rank["completed_users"] == rank["assigned_users"] == USERS
                and rank["execution_topology"] == "SINGLE_GPU" and rank["world_size"] == 1
                and rank["execution_profile"] == plan["execution_profile"]
                and rank["generation"]["num_beams"] == rank["generation"]["num_return_sequences"] == BEAMS
                and rank["generation"]["filter_items"] is False, (name, "rank scope and beam"))
        require({k: v["sha256"] for k, v in rank["data_identity"].items()} == plan["asset_sha256"],
                (name, "rank asset identity"))
        require(rank["assignment_sha256"] == json_digest(rank["assigned_row_indices"])
                and sorted(rank["assigned_row_indices"]) == list(range(len(users))), (name, "assignment digest"))
        require((completion / "original" / name / "test/rank0.json").read_bytes() == raw, (name, "delivered original rank bytes"))
        ranks[name], hashes[name] = rank, digest(raw)
    normal = ranks["normal"]
    token_map = normal["tokenizer"]["added_tokens"]
    require(set(token_map.values()) == set(range(32000, 32786)) and json_digest(token_map, sort=True) == TOKENIZER_MAP_PIN,
            "tokenizer SID content identity")
    for name in ("s3", "s4"):
        rank = ranks[name]
        for field in ("assigned_row_indices", "assignment_sha256", "tokenizer", "generation",
                      "tokenizer_identity", "allowed_callback_sha256", "adapter_files"):
            require(rank[field] == normal[field], (name, field, "common condition axis"))
        require(rank["shuffle"] == {"n": int(name[-1]), "seed": 42, "protocol": SHUFFLE_PROTOCOL},
                (name, "saved condition configuration"))
    off = normal["normal_off_same_process_exact"]
    require(len(off) == len(users) and {x["row_idx"] for x in off} == set(range(len(users)))
            and all(x["same_process_exact"] is True for x in off)
            and "--verify-normal-off" in normal["argv"], "complete original Normal/OFF evidence")
    return normal["assigned_row_indices"], token_map, hashes, len(off)


def review_predictions(test_dirs, users, inter, index, axis, token_map):
    sid_by_item = {key: "".join(value) for key, value in index.items()}
    sizes = Counter(sid_by_item.values())
    require(len(index) == len(sizes) == 12101, "real catalog and zero collisions")
    token_by_id = {v: k for k, v in token_map.items()}
    matrix = np.empty((3, USERS, len(KEYS)), dtype=np.float64)
    hashes = [hashlib.sha256() for _ in CONDITIONS]
    totals = [0, 0, 0]
    groups, moved_d, behavior = Counter(), Counter(), Counter()
    starts = [Counter() for _ in CONDITIONS]
    diagnostics = [{k: Counter() for k in K_VALUES} for _ in CONDITIONS]
    coverage = [{k: set() for k in K_VALUES} for _ in CONDITIONS]
    files = [p / "predictions.rank0.jsonl" for p in test_dirs]
    before = [p.stat() for p in files]
    with ExitStack() as stack:
        sources = [stack.enter_context(p.open("rb")) for p in files]
        count = 0
        for position, lines in enumerate(zip_longest(*sources)):
            require(position < USERS and all(lines), (position, "extra/missing paired raw rows"))
            all_rows = []
            for j, line in enumerate(lines):
                hashes[j].update(line)
                totals[j] += len(line)
                all_rows.append(json.loads(line))
            i = axis[position]
            user = users[i]
            interactions = inter[user]
            item = str(interactions[-1])
            target = sid_by_item[item]
            history = [index[str(value)] for value in interactions[:-1][-20:]]
            lists = []
            for j, (name, row) in enumerate(zip(CONDITIONS, all_rows, strict=True)):
                require((row["row_idx"], row["user_id"], row["target_item_id"], row["target"]) == (i, user, item, target)
                        and (row["condition"], row["scope"], row["rank"], row["world_size"], row["prompt_id"]) == (name, "FORMAL", 0, 1, 0)
                        and row["model_identity_sha256"] == MODEL_PIN, (name, i, "raw row identity / order"))
                group_count, moved = verify_model_input(name, row, all_rows[0], history, token_map)
                groups[name] += group_count
                moved_d[name] += moved
                candidates = row["candidates"]
                require(len(candidates) == BEAMS, (name, i, "beam count"))
                ordered_by_score = []
                for offset, c in enumerate(candidates):
                    score = c["sequence_score"]
                    ids = c["generated_token_ids"]
                    sid = c["raw_decoded"].rsplit("Response:", 1)[-1].strip().replace(" ", "")
                    require(c["generation_order"] == offset and math.isfinite(score) and
                            len(ids) >= 5 and ids[4] == 2 and c["eos_position"] == 4
                            and "".join(token_by_id[t] for t in ids[:4]) == sid == c["sid"],
                            (name, i, offset, "candidate token/decode/score agreement"))
                    ordered_by_score.append((-score, offset, sid))
                ranked = sorted(ordered_by_score)
                ordered = [sid for _, _, sid in ranked]
                require([offset for _, offset, _ in ranked] == row["sorted_generation_order"]
                        and [int(sid == target) for sid in ordered] == row["relevance"],
                        (name, i, "frozen relevance and stable ordering"))
                matrix[j, i], first = metric_vector(ordered, target, sizes)
                starts[j][str(first) if first else "miss"] += 1
                for k in K_VALUES:
                    subset = ordered[:k]
                    valid = sum(sid in sizes for sid in subset)
                    dup = k - len(set(subset))
                    coll = sum(sizes.get(sid, 0) > 1 for sid in subset)
                    diagnostics[j][k].update(candidates=k, valid=valid, invalid=k-valid, duplicates=dup,
                                             colliding=coll, users_invalid=int(valid < k),
                                             users_duplicate=int(dup > 0), users_colliding=int(coll > 0))
                    coverage[j][k].update(set(subset).intersection(sizes))
                lists.append(ordered)
            for name, shuffled in zip(CONDITIONS[1:], lists[1:], strict=True):
                behavior[name, "top1_changed"] += shuffled[0] != lists[0][0]
                behavior[name, "top20_ordered_changed"] += shuffled != lists[0]
                behavior[name, "top20_set_changed"] += set(shuffled) != set(lists[0])
            count += 1
    require(count == USERS, "full raw population")
    observed = {}
    for j, name in enumerate(CONDITIONS):
        after = files[j].stat()
        fields = ("st_ctime_ns", "st_mtime_ns", "st_size", "st_ino", "st_dev")
        require(all(getattr(before[j], f) == getattr(after, f) for f in fields)
                and (totals[j], hashes[j].hexdigest()) == PREDICTIONS[name], (name, "raw bytes/hash stability"))
        observed[name] = {"path": str(files[j]), "sha256": hashes[j].hexdigest(), "bytes": totals[j],
                          "users": count, "candidate_count": count * BEAMS,
                          "history_items": groups[name], "fourth_position_moved_items": moved_d[name],
                          "first_target_rank_users": {str(i): starts[j][str(i)] for i in range(1, 21)}
                                                     | {"miss": starts[j]["miss"]},
                          "diagnostics": {str(k): {**diagnostics[j][k], "unique_valid_sids": len(coverage[j][k])}
                                          for k in K_VALUES}}
    require(groups["normal"] == groups["s3"] == groups["s4"] and moved_d["s3"] == 0 < moved_d["s4"],
            "actual fourth-position scope")
    return matrix, behavior, observed


def review_offline(eval_dirs, completion, expected, observations, index, inter, users):
    rows_by_condition, matrix_meta = [], {}
    sid_by_item = {key: "".join(value) for key, value in index.items()}
    frequencies = Counter(str(item) for sequence in inter.values() for item in sequence[:-2])
    ordered = sorted(frequencies.values())
    q1, q2 = ordered[(len(ordered)+2)//3-1], ordered[(2*len(ordered)+2)//3-1]
    for j, (name, d) in enumerate(zip(CONDITIONS, eval_dirs, strict=True)):
        metrics = read_json(d / "metrics.json")
        require(metrics["status"] == "PASS" and metrics["users"] == USERS
                and metrics["official_cce_full_user_comparisons"] == USERS
                and metrics["user_axis_sha256"] == AXIS_PIN and list(metrics["metrics"]) == KEYS
                and metrics["prediction_files"][0]["sha256"] == PREDICTIONS[name][1], (name, "production summary identity"))
        mirrored = completion / "original" / name / "primary"
        names = ("metrics.json", "per_user.csv", "per_user_metrics.npz", "groups.json", "examples.json",
                 "bootstrap.json", "bootstrap_draw_means.npy", "unit_tests.json")
        original_hashes = {}
        for filename in names:
            source = d / filename
            raw = source.read_bytes()
            require((mirrored / filename).read_bytes() == raw, (name, filename, "actual original and closure copy"))
            original_hashes[filename] = digest(raw)
        with np.load(d / "per_user_metrics.npz", allow_pickle=False) as archive:
            require(archive["keys"].tolist() == KEYS and archive["user_axis_sha256"].item() == AXIS_PIN
                    and archive["row_idx"].tolist() == list(range(USERS)), (name, "original matrix axis"))
            actual = archive["values"]
        require(actual.dtype == np.float64 and actual.shape == (USERS, len(KEYS))
                and np.isfinite(actual).all(), (name, "original matrix storage"))
        with (d / "per_user.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        require(len(csv_rows) == USERS, (name, "CSV population"))
        csv_value = np.array([[float(row[k]) for k in KEYS] for row in csv_rows], dtype=np.float64)
        require(np.array_equal(actual, csv_value), (name, "exact original CSV and NPZ values"))
        for i, row in enumerate(csv_rows):
            seq = inter[users[i]]
            item = str(seq[-1])
            frequency = frequencies[item]
            h = min(len(seq)-1, 20)
            group = "0-5" if h <= 5 else "6-10" if h <= 10 else "11-20"
            popularity = "UNSEEN" if frequency == 0 else "LONG_TAIL" if frequency <= q1 else "MID" if frequency <= q2 else "POPULAR"
            require(tuple(row[k] for k in PAIR_FIELDS) ==
                    (str(i), users[i], AXIS_PIN, item, sid_by_item[item], "1", str(len(seq)), str(h),
                     group, str(frequency), popularity, "letter_beauty_beam20_v1"),
                    (name, i, "original paired metadata from real data"))
        first_counts = Counter(str(row["first_target_rank"]) if int(row["first_target_rank"]) else "miss" for row in csv_rows)
        report_counts = {str(r): first_counts[str(r)] for r in range(1, 21)} | {"miss": first_counts["miss"]}
        require(report_counts == observations[name]["first_target_rank_users"] == metrics["first_target_rank_users"],
                (name, "first target histogram against CSV and metrics"))
        for k, values in observations[name]["diagnostics"].items():
            require(all(v == metrics["sid_statistics"][k][field] for field, v in values.items()), (name, k, "SID diagnostics"))
        max_errors = np.max(np.abs(actual - expected[j]), axis=0)
        require(np.all(max_errors < 1e-13) and np.all(max_errors[10:] <= 1e-14)
                and np.array_equal(actual[:, :len(HIT_K)], expected[j, :, :len(HIT_K)]),
                (name, "new independent metric matrix compared against original"))
        summary = {key: math.fsum(expected[j, :, i]) / USERS for i, key in enumerate(KEYS)}
        mean_difference = max(abs(summary[k] - metrics["metrics"][k]) for k in KEYS)
        require(mean_difference < 1e-13, (name, "original primary summary"))
        review_file = completion / "independent" / f"{name}.npz"
        with np.load(review_file, allow_pickle=False) as produced:
            require(produced["keys"].tolist() == KEYS and produced["row_idx"].tolist() == list(range(USERS)),
                    (name, "new producer matrix axis"))
            replay = produced["values"]
        require(replay.dtype == np.float64 and replay.shape == (USERS, len(KEYS)) and np.isfinite(replay).all(),
                (name, "new producer matrix dtype/shape"))
        raw_replay_delta = pair_delta(replay, expected[j])
        require(raw_replay_delta < 1e-13 and pair_delta(replay[:, 10:], expected[j, :, 10:]) <= 1e-14
                and np.array_equal(replay[:, :len(HIT_K)], expected[j, :, :len(HIT_K)]),
                (name, "new producer independent comparison"))
        report_file = completion / "independent" / f"{name}.json"
        report = read_json(report_file)
        require(report["status"] == "PASS" and report["condition"] == name and report["run_commit"] == RUN_PIN
                and report["prediction"]["sha256"] == PREDICTIONS[name][1]
                and report["first_target_rank_users"] == report_counts
                and all(abs(report["metrics"][key] - summary[key]) < 1e-13 for key in KEYS),
                (name, "producer independent metric report"))
        matrix_meta[name] = {"original_present": original_hashes, "max_per_metric_error_vs_original_npz": dict(zip(KEYS, max_errors.tolist())),
                             "mean_max_error": mean_difference, "closure_independent_matrix_sha256": digest(review_file.read_bytes()),
                             "closure_independent_max_error": float(raw_replay_delta), "mean": summary,
                             "closure_independent_report_sha256": payload_identity(report_file)["sha256"]}
        rows_by_condition.append([tuple(row[k] for k in PAIR_FIELDS) for row in csv_rows])
    require(rows_by_condition[0] == rows_by_condition[1] == rows_by_condition[2], "exact paired CSV metadata")
    return matrix_meta


def reviewers_draws(difference):
    # Column-wise with an independently chosen block size. The draw stream is
    # replicate-major and the seed resets for each condition as in production.
    rng = np.random.Generator(np.random.PCG64(20261113))
    draws = np.empty((5000, len(KEYS)), dtype=np.float64)
    columns = difference.T.copy()
    for start in range(0, 5000, 11):
        index = rng.integers(0, USERS, size=(min(11, 5000-start), USERS), dtype=np.int64)
        for k, values in enumerate(columns):
            draws[start:start+len(index), k] = values[index].mean(axis=1)
    return draws


def review_pairing(completion, expected, behavior, matrix_meta, old_pair):
    outputs = {}
    for j, name in enumerate(CONDITIONS[1:], 1):
        folder = completion / "paired" / name
        paired_path = folder / "paired_comparison.json"
        paired = read_json(paired_path)
        draws_path = folder / "paired_bootstrap_difference_means.npy"
        actual_draws = np.load(draws_path, allow_pickle=False)
        require(actual_draws.shape == (5000, len(KEYS)) and actual_draws.dtype == np.float64
                and np.isfinite(actual_draws).all(), (name, "paired outputs shape/finite"))
        require(paired["users"] == USERS and paired["status"] == "PASS" and list(paired["metrics"]) == KEYS
                and paired["row_indices"] == list(range(USERS)), (name, "paired output identities"))
        protocol = paired["bootstrap"]
        require((protocol["repetitions"], protocol["seed"], protocol["rng"], protocol["replacement"], protocol["percentile"]) ==
                (5000, 20261113, "PCG64", True, "95% linear"), (name, "paired protocol"))
        expected_draws = reviewers_draws(expected[j] - expected[0])
        difference = pair_delta(actual_draws, expected_draws)
        interval = np.percentile(expected_draws, [2.5, 97.5], axis=0, method="linear")
        supplied_interval = np.array([[paired["metrics"][k][field] for k in KEYS]
                                      for field in ("difference_CI95_low", "difference_CI95_high")])
        ci_error = pair_delta(interval, supplied_interval)
        require(difference <= 1e-13 and ci_error <= 1e-13, (name, "paired draw and interval verification"))
        for i, key in enumerate(KEYS):
            n, s = matrix_meta["normal"]["mean"][key], matrix_meta[name]["mean"][key]
            record = paired["metrics"][key]
            require(all(abs(record[metric] - value) <= 1e-13
                        for metric, value in (("Normal", n), ("Shuffle", s), ("absolute_difference", s-n),
                                              ("relative_difference", (s-n)/n))), (name, key, "paired metric means"))
            if key.startswith("hit@"):
                transition = {f"{a}->{b}": int(((expected[0, :, i] == a) & (expected[j, :, i] == b)).sum())
                              for a in (0, 1) for b in (0, 1)}
                require(transition == paired["changes"]["hit_transitions"][key]
                        and sum(transition.values()) == USERS
                        and transition["1->0"] + transition["1->1"] == int(expected[0, :, i].sum())
                        and transition["0->1"] + transition["1->1"] == int(expected[j, :, i].sum()),
                        (name, key, "hit transitions from raw inputs"))
        for key in ("top1_changed", "top20_ordered_changed", "top20_set_changed"):
            require(paired["changes"][key] == behavior[name, key]
                    and paired["changes"][key+"_fraction"] == behavior[name, key] / USERS,
                    (name, key, "ordered and unordered rank changes"))
        if name == "s3":
            old_raw = old_pair / "paired_comparison.json"
            old_draws = old_pair / "paired_bootstrap_difference_means.npy"
            require(digest(old_raw.read_bytes()) == "6899948326c5ef5ab10aca80481bad6c7d1bdbfa6b708e3e7745aefe4260d29c" and
                    digest(old_draws.read_bytes()) == "b176dfda7bc00244b332a46ccb8e34920681be761de9c55c5e56bf83c827f297",
                    "original isolated S3 timeout artifacts")
            original = read_json(old_raw)
            require(original["changes"] == paired["changes"] and original["bootstrap"] == paired["bootstrap"],
                    "original S3 paired changes and protocol")
            require(pair_delta(np.load(old_draws, allow_pickle=False), actual_draws) <= 1e-13, "original S3 full draws")
            for key in KEYS:
                require(all(abs(value - paired["metrics"][key][field]) <= 1e-13
                            for field, value in original["metrics"][key].items()), (key, "original S3 metrics"))
        outputs[name] = {"saved_report": payload_identity(paired_path), "stored_draws": payload_identity(draws_path),
                         "reviewer_draw_max_error": float(difference), "reviewer_interval_max_error": float(ci_error),
                         "bootstrap": protocol, "changes": paired["changes"]}
    return outputs


def review_delivery(completion):
    manifest_path = completion / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["kind"] == "compute" and manifest["original_run_commit"] == RUN_PIN,
            "completion delivery manifest identity")
    file_set = {str(p.relative_to(completion)) for p in completion.rglob("*") if p.is_file()} - {"manifest.json"}
    require(file_set == set(manifest["files"]), "completion manifest members")
    for relative, recorded in manifest["files"].items():
        path = completion / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.resolve(strict=True).is_relative_to(completion) and not path.is_symlink(),
                (relative, "delivery manifest path"))
        actual = payload_identity(path)
        require(all(actual[key] == recorded[key] for key in ("bytes", "sha256")), (relative, "delivery bytes"))
    refs = {name: read_json(completion / "paired" / name / "paired_comparison.json")["metrics"]
            for name in CONDITIONS[1:]}
    table_path = completion / "condition_metrics.csv"
    with table_path.open() as stream:
        rows = list(csv.DictReader(stream))
    require([row["metric"] for row in rows] == KEYS, "completion table metric axis")
    for row in rows:
        s3, s4 = (refs[name][row["metric"]] for name in CONDITIONS[1:])
        expected = {"Normal": s3["Normal"], "S3": s3["Shuffle"], "S4": s4["Shuffle"]}
        for name, pair in (("S3", s3), ("S4", s4)):
            for column, field in (("minus_Normal", "absolute_difference"), ("relative_to_Normal", "relative_difference"),
                                  ("CI95_low", "difference_CI95_low"), ("CI95_high", "difference_CI95_high")):
                expected[f"{name}_{column}"] = pair[field]
        require(all(float(row[name]) == value for name, value in expected.items()),
                (row["metric"], "condition table versus paired outputs"))
    return {"file_members_verified": len(file_set), "manifest": payload_identity(manifest_path),
            "condition_table": payload_identity(table_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--completion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    tick = time.monotonic()
    config = read_json(args.config)
    root = Path(config["original_root"]).resolve(strict=True)
    completion = args.completion.resolve(strict=True)
    output = args.output.resolve(strict=True)
    require(not output.is_relative_to(root) and not output.is_relative_to(completion)
            and not any(output.iterdir()), "review output must be isolated and empty")
    plan_path = root / "plan.json"
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    canonical = json_digest(plan, sort=True)
    require(digest(plan_raw) == PLAN_PIN and canonical == CANON_PIN
            and plan["run_commit"] == config["run_commit"] == RUN_PIN, "sealed original plan and commit")
    state_path = root / "state.json"
    state_raw = state_path.read_bytes()
    state = json.loads(state_raw)
    paired_path, previous = stage_output(state, root, "paired")
    require(previous["runner"]["status"] == "TIMED_OUT" and previous["runner"]["exit_code"] == -15
            and previous["runner"]["cleanup_verified"] and "collect" not in state["stages"],
            "historical failure/cleanup stays distinct from supplemental work")
    completed = ("train", "verify_best_adapter", "normal", "normal_eval", "s3", "s3_eval", "s4", "s4_eval")
    for name in completed:
        _, attempt = stage_output(state, root, name)
        runner = attempt["runner"]
        require((runner["status"], runner["exit_code"], runner["cleanup_verified"]) == ("SUCCEEDED", 0, True)
                and state["stages"][name]["status"] == "COMPLETE", (name, "original cleared stage"))
    data = {}
    for key, sha in plan["asset_sha256"].items():
        path = Path(plan["data_root"]) / "Beauty" / f"Beauty.{key}.json"
        raw = path.read_bytes()
        require(digest(raw) == sha, (str(path), "frozen original data"))
        data[key] = json.loads(raw)
    index, inter = data["index"], data["inter"]
    users = list(inter)
    require(len(users) == USERS and json_digest(users) == AXIS_PIN
            and set(index) == set(data["item"]), "real data axes")
    test_dirs = [stage_output(state, root, name)[0] / "test" for name in CONDITIONS]
    eval_dirs = [stage_output(state, root, name + "_eval")[0] / "primary" for name in CONDITIONS]
    axis, token_map, rank_hashes, off_count = review_ranks(test_dirs, users, plan, completion)
    times = {"identity_seconds": time.monotonic()-tick}
    print("Non-author sealed inputs and Normal/OFF receipts verified; streaming formal predictions", flush=True)
    part = time.monotonic()
    expected, behavior, observations = review_predictions(test_dirs, users, inter, index, axis, token_map)
    times["full_raw_prediction_seconds"] = time.monotonic()-part
    print("Non-author raw streams verified; comparing original CSV/NPZ and closure matrices", flush=True)
    part = time.monotonic()
    matrix = review_offline(eval_dirs, completion, expected, observations, index, inter, users)
    times["offline_files_seconds"] = time.monotonic()-part
    print("Non-author matrices verified; regenerating both paired bootstrap streams", flush=True)
    part = time.monotonic()
    pairs = review_pairing(completion, expected, behavior, matrix, paired_path / "s3")
    times["paired_draws_seconds"] = time.monotonic()-part
    part = time.monotonic()
    delivery = review_delivery(completion)
    times["completion_delivery_seconds"] = time.monotonic()-part
    require(plan_path.read_bytes() == plan_raw and state_path.read_bytes() == state_raw, "original identity remained stable")
    report = {"status": "PASS", "scope": "NON_AUTHOR_DIRECT_ORIGINAL_AND_COMPLETION",
              "completion_evidence": payload_identity(completion/"completion.json"),
              "review_source": payload_identity(Path(__file__)), "review_config": payload_identity(args.config),
              "run_commit": RUN_PIN, "plan_raw_sha256": PLAN_PIN, "plan_canonical_sha256": CANON_PIN,
              "original_state_sha256": digest(state_raw), "original_completed_stages_verified": list(completed),
              "asset_sha256": plan["asset_sha256"], "user_axis_sha256": AXIS_PIN,
              "rank_report_sha256": rank_hashes, "normal_off_exact_count": off_count,
              "source_predictions": observations, "metric_matrices": matrix, "paired": pairs, "delivery": delivery,
              "old_paired_job_id": previous["job_id"], "elapsed_seconds": time.monotonic()-tick, "timings": times}
    with (output / "review_checks.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Non-author reviewed {USERS} users from each raw prediction file in {report['elapsed_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
