"""V2 draw and GRID's level-offset SID boundary."""
import hashlib
import json
from itertools import permutations
from functools import lru_cache

from shuffle_reference import relative as reference_draw, remap as reference_remap

PROTOCOL = "GRID-LETTER-SHUFFLE-V2"
SHUFFLE_SEED = 42
LEVELS = {"Normal": (), "Shuffle3a": (0, 1, 2), "Shuffle4": (0, 1, 2, 3)}
IDENTITY = (0, 1, 2, 3)
WIDTH = 120
SLOTS = 256


@lru_cache(maxsize=2)
def catalog(m):
    if m not in (3, 4):
        raise ValueError(f"Invalid active width {m}")
    return tuple(permutations(range(m)))


def draw(m, row, occurrence, digest=hashlib.sha256):
    """Pure subset of frozen V2 core.draw with injectable hash for the rejection fixture."""
    if any(type(n) is not int or n < 0 for n in (m, row, occurrence)):
        raise ValueError("V2 coordinates must be nonnegative integers")
    choices = catalog(m)
    limit = ((1 << 256) // len(choices)) * len(choices)
    counter = 0
    while True:
        payload = json.dumps([PROTOCOL, SHUFFLE_SEED, m, row, occurrence, counter],
                             ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        value = int.from_bytes(digest(payload).digest(), "big")
        if value < limit:
            index = value % len(choices)
            return index, choices[index], counter
        counter += 1


def source_positions(mode, row, item_count):
    if type(row) is not int or row < 0 or not 0 <= item_count <= 20:
        raise ValueError("Evaluation row or visible history length out of range")
    active = LEVELS[mode]
    for occurrence in range(item_count):
        full = list(IDENTITY)
        index, rejected = 0, 0
        if active:
            index, relative, rejected = draw(len(active), row, occurrence)
            for position, source in zip(active, relative, strict=True):
                full[position] = active[source]
        yield tuple(full), index, rejected


def history_tokens(history, codes):
    return [code for item in history for code in codes[item]]


def verify_input(values, mask, history, codes):
    expected = history_tokens(history, codes)
    assert len(history) <= 20 and len(values) == len(mask) == WIDTH
    assert values == expected + [-1] * (WIDTH - len(expected)), "Unexpected native evaluation SID boundary"
    assert mask == [1] * len(expected) + [0] * (WIDTH - len(expected)), "Unexpected native evaluation mask"


def shifted(values, mask):
    return [(value + (i % 4) * SLOTS) * bit for i, (value, bit) in enumerate(zip(values, mask, strict=True))]


def expected_tokens(before, history_length, draws):
    result = list(before)
    for i, (perm, _, _) in enumerate(draws):
        start = 4 * i
        result[start:start + 4] = [before[start + j] for j in perm]
    assert len(draws) == history_length
    return result


def cpu_replay(sequences, row_map, codes=None):
    """Replay every fixed validation and test history using independent factoradic reference."""
    import collections
    if set(row_map) != set(range(len(sequences))) or set(row_map.values()) != set(row_map):
        raise ValueError("V2 row axis differs from the complete evaluation user axis")
    results = {}
    for split in ("evaluation", "testing"):
        for mode, active in LEVELS.items():
            counts = collections.Counter()
            frequencies = collections.Counter()
            digest = hashlib.sha256()
            for uid, sequence in enumerate(sequences):
                row = row_map[uid]
                history = sequence[-21:-1] if split == "testing" else sequence[:-1][-21:-1]
                target = sequence[-1 if split == "testing" else -2]
                plans = list(source_positions(mode, row, len(history)))
                counts["users"] += 1
                counts["groups"] += len(history)
                counts["padding_positions"] += WIDTH - 4 * len(history)
                for j, (full, index, rejected) in enumerate(plans):
                    a, b, c, d = codes[history[j]] if codes is not None else (7, 7, 101, 3)
                    tokens = [a, 256 + b, 512 + c, 768 + d]
                    if active:
                        ref_index, permutation, ref_rejected = reference_draw(len(active), row, j)
                        assert (index, rejected) == (ref_index, ref_rejected)
                        expect = reference_remap(tokens, active, permutation)
                    else:
                        expect = tokens
                    actual = [tokens[k] for k in full]
                    assert actual == expect and sorted(actual) == tokens
                    assert all(actual[k] == tokens[k] for k in IDENTITY if k not in active)
                    counts["drawn_nonidentity"] += int(index != 0)
                    counts["tokens_changed"] += int(actual != tokens)
                    counts["rejected_draws"] += rejected
                    frequencies[index] += 1
                digest.update(json.dumps([row, history, target, plans], separators=(",", ":")).encode() + b"\n")
            # Independently traverse reversed irregular shards and require each absolute coordinate
            # to match the reference again. Include shard checks for Normal as well.
            for end in range(len(sequences), 0, -137):
                for uid in reversed(range(max(0, end - 137), end)):
                    sequence = sequences[uid] if split == "testing" else sequences[uid][:-1]
                    length = min(len(sequence) - 1, 20)
                    plans = list(source_positions(mode, row_map[uid], length))
                    for j, (full, index, rejected) in enumerate(plans):
                        if active:
                            ri, rp, rr = reference_draw(len(active), row_map[uid], j)
                            assert (index, rejected, list(full)) == (ri, rr, reference_remap(list(IDENTITY), active, rp))
                        else:
                            assert (full, index, rejected) == (IDENTITY, 0, 0)
            assert len(frequencies) == (len(catalog(len(active))) if active else 1)
            results[f"{split}/{mode}"] = dict(counts, frequencies=dict(frequencies),
                                             canonical_replay_sha256=digest.hexdigest(),
                                             reverse_shards_verified=True)
    return results


def fixtures():
    import itertools
    class SyntheticHash:
        def __init__(self, value):
            self.value = value
        def digest(self):
            return self.value.to_bytes(32, "big")
        def hexdigest(self):
            return self.digest().hex()

    for mode in ("Shuffle3a", "Shuffle4"):
        active = LEVELS[mode]
        m = len(active)
        tokens = [5, 261, 517, 773]
        for i, p in enumerate(itertools.permutations(active)):
            def mock_hash(payload):
                return SyntheticHash((1 << 256) - 1 if payload.endswith(b",0]") else i)
            index, sampled, rejected = draw(m, 0, 0, mock_hash)
            assert (index, sampled, rejected) == (i, p, 1)
            ri, rp, rr = reference_draw(m, 0, 0, mock_hash)
            assert (index, list(sampled), rejected) == (ri, rp, rr)
            expected = [tokens[k] for k in (*p, *range(m, 4))]
            assert reference_remap(tokens, active, p) == expected
    for args in ((2, 0, 0), (3, -1, 0), (4, 3, -1), (3, True, 0)):
        try:
            draw(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(args)
    codes = [(5, 5, 5, 5)]
    values = [5] * 4 + [-1] * 116
    mask = [1] * 4 + [0] * 116
    verify_input(values, mask, [0], codes)
    for invalid in ([1] * 3 + [0] * 117, [0] * 4 + [1] * 4 + [0] * 112):
        try:
            verify_input(values, invalid, [0], codes)
        except AssertionError:
            pass
        else:
            raise AssertionError("partial or misaligned item passed")
    return {"all_permutation_ids": [6, 24], "forced_rejection": True,
            "equal_bare_codes_distinct_levels": True, "invalid_boundaries_rejected": True}
