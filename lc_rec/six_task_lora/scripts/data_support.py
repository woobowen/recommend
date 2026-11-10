"""Small, CPU-only helpers for the six-task LC-Rec workload audit.

These helpers describe the original records; they do not edit source data,
replace official Dataset implementations, or define a training sampler.
"""
from __future__ import annotations

import bisect
import collections
import hashlib
import json
from pathlib import Path


TASKS = ["seqrec", "item2index", "index2item", "fusionseqrec", "itemsearch", "preferenceobtain"]
EXPECTED_ROWS = [131837, 9922, 9922, 20000, 24772, 24772]
EXPECTED_TOKENIZER_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def largest_remainder(counts, size):
    """Deterministic nearest-proportion allocation; task order breaks ties."""
    total = sum(counts)
    floor = [count * size // total for count in counts]
    order = sorted(range(len(counts)), key=lambda i: (-(counts[i] * size % total), i))
    for i in order[:size - sum(floor)]:
        floor[i] += 1
    return floor


def number_summary(values, np):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    quantiles = np.percentile(a, [10, 50, 90, 99], method="linear")
    return {"count": len(a), "min": float(a.min()), "p10": float(quantiles[0]),
            "p50": float(quantiles[1]), "p90": float(quantiles[2]), "p99": float(quantiles[3]),
            "max": float(a.max()), "mean": float(a.mean()), "variance": float(a.var()),
            "std": float(a.std()), "sum": float(a.sum())}


def padding_summary(lengths, batch_size, np, sort=False):
    """One rank, no drop_last; diagnostic layout, not a Trainer claim."""
    a = np.asarray(lengths, dtype=np.int64)
    if sort:
        a = np.sort(a, kind="stable")
    useful, padded, batches = 0, 0, 0
    for start in range(0, len(a), batch_size):
        part = a[start:start + batch_size]
        useful += int(part.sum())
        padded += int(part.max()) * len(part)
        batches += 1
    return {"batch_size_per_rank": batch_size, "batches": batches, "useful_tokens": useful,
            "allocated_token_slots": padded, "padding_tokens": padded - useful,
            "padding_ratio": 1 - useful / padded if padded else 0.0,
            "length_sorted_diagnostic": sort}


class ObservedRandom:
    """Transparent observation facade for data.py's random module reference.

    Every call delegates once to the original Python random module. No seed,
    selection, or RNG-consumption change is introduced. This facade is attached
    to the imported module object only; original source files remain untouched.
    """

    def __init__(self, original):
        self.original = original
        self.events = []

    def reset(self):
        self.events = []

    def randint(self, a, b):
        result = self.original.randint(a, b)
        self.events.append({"kind": "randint", "a": a, "b": b, "value": result})
        return result

    def choice(self, values):
        result = self.original.choice(values)
        self.events.append({"kind": "choice", "population": len(values),
                            "matching_value_indices": [i for i, value in enumerate(values) if value == result]})
        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


class Origins:
    """Bind official processed rows back to unchanged raw Instruments files."""

    def __init__(self, loaded, datasets, args, np):
        self.loaded, self.datasets, self.args = loaded, datasets, args
        self.inter = loaded["Instruments.inter.json"]
        self.index = loaded["Instruments.index.json"]
        self.item = loaded["Instruments.item.json"]
        self.user = loaded["Instruments.user.json"]
        self.uids = list(self.inter)
        self.explicit_uids = list(self.user["user_explicit_preference"])
        self.item_ids = list(self.item)
        self.cumulative, total = [], 0
        for sequence in self.inter.values():
            total += len(sequence) - 3
            self.cumulative.append(total)
        # In this pinned data.py, the first NumPy draw during load_datasets is
        # FusionSeqRec's choice: earlier datasets have sample_num=0. Validate
        # every reconstructed fusion row below rather than trusting this fact.
        fusion_rng = np.random.RandomState(args.seed)
        self.fusion_source_indices = fusion_rng.choice(total, 20000, replace=False).tolist()
        self.aux_status = {}
        self.aux_counts = collections.Counter()
        train = self.user["user_vague_intention"]["train"]
        test = self.user["user_vague_intention"].get("test", {})
        self.test_mismatch_count = 0
        self.test_checked = len(test)
        for uid, record in test.items():
            seq = list(map(str, self.inter.get(uid, [])))
            if not seq or str(record.get("item")) != seq[-1] or list(map(str, record.get("inters", []))) != seq[:-1]:
                self.test_mismatch_count += 1
        for uid in self.explicit_uids:
            seq = list(map(str, self.inter[uid]))
            record = train[uid]
            preferences = self.user["user_explicit_preference"][uid]
            history = list(map(str, record.get("inters", [])))
            target = str(record.get("item"))
            target_ok, history_ok = target == seq[-3], history == seq[:-3]
            querys = record.get("querys")
            text_ok = (isinstance(preferences, list) and bool(preferences)
                       and all(isinstance(x, str) and bool(x.strip()) for x in preferences)
                       and isinstance(querys, list) and len(querys) == 2
                       and all(isinstance(x, str) and bool(x.strip()) for x in querys))
            matching = [i for i, iid in enumerate(seq) if iid == target and seq[:i] == history]
            exact = target_ok and history_ok and text_ok
            self.aux_status[uid] = {"target_matches_current_train_minus3": target_ok,
                                    "history_matches_current_train_before_minus3": history_ok,
                                    "text_fields_valid": text_ok,
                                    "strict_train_pairing_match": exact,
                                    "aux_target_matching_full_sequence_positions": matching,
                                    "aux_target_is_current_validation_position": len(seq) - 2 in matching,
                                    "aux_target_is_current_test_position": len(seq) - 1 in matching}
            self.aux_counts["checked_users"] += 1
            self.aux_counts["strict_pairing_match" if exact else "mismatch"] += 1
            if len(seq) - 2 in matching:
                self.aux_counts["validation_position_matches"] += 1
            if len(seq) - 1 in matching:
                self.aux_counts["test_position_matches"] += 1

    def sid(self, iid):
        return "".join(self.index[str(iid)])

    def history_text(self, history):
        sid_history = [self.sid(iid) for iid in history]
        if self.args.add_prefix:
            sid_history = [str(i + 1) + ". " + sid for i, sid in enumerate(sid_history)]
        return self.args.his_sep.join(sid_history)

    def seq_origin(self, source_index):
        user_pos = bisect.bisect_right(self.cumulative, source_index)
        previous = self.cumulative[user_pos - 1] if user_pos else 0
        target_pos = source_index - previous + 1
        uid = self.uids[user_pos]
        sequence = list(map(str, self.inter[uid]))
        history = sequence[max(0, target_pos - self.args.max_his_len):target_pos]
        return {"user_id": uid, "source_training_prefix_index": int(source_index),
                "target_position_in_full_interaction": target_pos, "target_item_id": sequence[target_pos],
                "target_canonical_sid": self.sid(sequence[target_pos]), "history_item_ids": history}

    def at(self, task, index):
        ds = self.datasets[task]
        row_index = index // ds.prompt_sample_num
        if task in ("seqrec", "fusionseqrec"):
            source_index = self.fusion_source_indices[row_index] if task == "fusionseqrec" else row_index
            origin = self.seq_origin(source_index)
            actual = ds.inter_data[row_index]
            assert actual["item"] == origin["target_canonical_sid"]
            assert actual["inters"] == self.history_text(origin["history_item_ids"])
            if task == "fusionseqrec":
                iid = origin["target_item_id"]
                title = self.item[iid]["title"].strip().strip(".!?,;:`")
                assert actual["title"] == title
                assert actual["description"] == self.item[iid]["description"]
            origin["training_period_status"] = "verified_original_training_prefix"
            return origin, True
        if task in ("item2index", "index2item"):
            iid = self.item_ids[row_index]
            assert ds.feat_data[row_index]["item"] == self.sid(iid)
            return {"item_id": iid, "canonical_sid": self.sid(iid),
                    "training_period_status": "official_catalog_task_no_heldout_interaction_target"}, True
        uid = self.explicit_uids[row_index]
        raw = self.user["user_vague_intention"]["train"][uid]
        current = list(map(str, self.inter[uid]))
        pairing = self.aux_status[uid]
        if task == "itemsearch":
            history = list(map(str, raw["inters"]))[-self.args.max_his_len:]
            assert ds.search_data[row_index]["inters"] == self.history_text(history)
            assert ds.search_data[row_index]["item"] == self.sid(raw["item"])
        else:
            history = current[:-3][-self.args.max_his_len:]
            assert ds.preference_data[row_index]["inters"] == self.history_text(history)
        origin = {"user_id": uid, "history_item_ids": history,
                  "intention_target_item_id": str(raw["item"]), "intention_target_canonical_sid": self.sid(raw["item"]),
                  "auxiliary_pairing": pairing,
                  "training_period_status": "strict_current_training_pairing_supports_technical_use"
                  if pairing["strict_train_pairing_match"] else "diagnostic_only_auxiliary_pairing_mismatch",
                  "explicit_preference_provenance_limit":
                  "Official generation code uses history[:-3], but per-record preference-generation inputs are not archived here. Pairing is evidence for technical use, not proof of a clean formal training corpus."}
        return origin, pairing["strict_train_pairing_match"]

    def summary(self):
        return {"train_auxiliary_pairing": dict(self.aux_counts), "test_auxiliary_checked_users": self.test_checked,
                "test_auxiliary_mismatch_users": self.test_mismatch_count,
                "affected_tasks": ["itemsearch", "preferenceobtain"],
                "unaffected_direct_split_tasks": ["seqrec", "item2index", "index2item", "fusionseqrec"],
                "formal_training_status": "BLOCKED_UNRESOLVED_AUXILIARY_PROVENANCE_AND_SPLIT_PAIRING",
                "technical_sample_rule": "For auxiliary tasks require exact unmodified train intention target==inter[-3], history==inter[:-3], and valid original text fields. This is diagnostic selection only, never a proposed formal-data filter.",
                "required_author_material": ["Interaction split/remapping and SID identity matching the released auxiliary records",
                                             "Per-user input histories or generation manifest for explicit preferences and intention queries",
                                             "Author-confirmed original split/preprocessing recipe; no regenerated or edited labels in this task"]}
