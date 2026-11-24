"""Check edge histories with the native CPU collator."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

from data_assets import official, sample_users

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "native"))


def check(root):
    import torch
    from src.data.loading.components.collate_functions import collate_fn_train, collate_with_sid_causal_duplicate
    from src.data.loading.components.label_function import NextKTokenMasking

    codes, sequences, groups = official(root)
    uids = sample_users(codes, sequences, groups)
    labels = {"sequence_data": SimpleNamespace(transform=NextKTokenMasking(next_k=4))}
    options = {"labels": labels, "sequence_length": 120, "padding_token": -1, "masking_token": -1}

    def raw(u, sequence):
        return {"sequence_data": torch.tensor([v for i in sequence for v in codes[i]]),
                "user_id": torch.tensor([u])}

    eval_lengths = []
    for start in range(0, len(uids), 8):
        users = uids[start:start + 8]
        slices = [sequences[u][-21:] for u in users]
        batch = [raw(u, seq) for u, seq in zip(users, slices)]
        inputs, target = collate_fn_train(batch, **options)
        assert inputs.transformed_sequences["user_id"][:, 0].tolist() == users
        assert target.labels["sequence_data"].reshape(-1, 4).tolist() == [list(codes[s[-1]]) for s in slices]
        for i, seq in enumerate(slices):
            history = [v for item in seq[:-1] for v in codes[item]]
            assert inputs.transformed_sequences["sequence_data"][i].tolist() == history + [-1] * (120 - len(history))
            assert inputs.mask[i].tolist() == [1] * len(history) + [0] * (120 - len(history))
            eval_lengths.append(len(history) // 4)
    user = max(range(len(sequences)), key=lambda u: len(sequences[u]))
    sequence = sequences[user][:-2]
    # Independent enumerate all contiguous windows; the source trims each
    # selected window only after sampling. Match actual histories + labels.
    valid = {tuple(sequence[start:end][-30:]) for end in range(2, len(sequence) + 1)
             for start in range(end - 1)}
    valid_codes = {tuple(v for i in seq for v in codes[i]) for seq in valid}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        inputs, target = collate_with_sid_causal_duplicate([raw(user, sequence)], "sequence_data", 4,
                                                             max_batch_size=128, **options)
    observations = []
    for row, mask, y in zip(inputs.transformed_sequences["sequence_data"], inputs.mask,
                            target.labels["sequence_data"].reshape(-1, 4)):
        length = int(mask.sum())
        assert 4 <= length <= 116 and length % 4 == 0
        assert tuple(row[:length].tolist() + y.tolist()) in valid_codes
        assert all(row[length:] == -1)
        observations.append(length // 4 + 1)
    assert len(observations) <= 128 and max(observations) == 30
    return {"test_users": uids, "test_batch_sizes": [8, 8, 4], "test_min_history": min(eval_lengths),
            "test_max_history": max(eval_lengths), "full_training_items_for_one_user": len(sequence),
            "training_fixture_expanded": len(observations), "max_items_with_target": max(observations)}


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])), indent=2))
