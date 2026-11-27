"""CPU-only replay against the historical LETTER Beauty sampler and input log."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rows_sha(rows):
    return digest_bytes(json.dumps(rows, separators=(",", ":")).encode())


def sampler_replay(torch, batch_sampler_shard, n, initial, sampling):
    start = initial.get_state()
    result = []
    for rank in range(2):
        generator = torch.Generator()
        generator.set_state(start)
        sampler = sampling.RandomSampler(range(n), replacement=False, generator=generator)
        batches = sampling.BatchSampler(sampler, batch_size=8, drop_last=False)
        shard = batch_sampler_shard(batches, num_processes=2, process_index=rank,
                                    split_batches=False, even_batches=True)
        result.append([index for batch in shard for index in batch])
        if rank == 0:
            initial.set_state(generator.get_state())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "config", "data-root", "model", "historical-training"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import torch
    from torch.utils import data as sampling
    from accelerate.data_loader import BatchSamplerShard
    from transformers import LlamaTokenizer
    from utils import load_datasets, set_seed
    from prompt import all_prompt
    from collator import Collator

    config_bytes = args.config.read_bytes()
    if digest_bytes(config_bytes) != "24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd":
        raise ValueError("historical config mismatch")
    cfg = SimpleNamespace(**json.loads(config_bytes))
    cfg.data_path = str(args.data_root)
    set_seed(cfg.seed)
    train, valid = load_datasets(cfg)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model, model_max_length=2048, padding_side="right", local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokens = train.datasets[0].get_new_tokens()
    if tokenizer.add_tokens(tokens) != 965 or len(tokenizer) != 32965:
        raise ValueError("historical vocabulary mismatch")
    old_tokenizer = LlamaTokenizer.from_pretrained(
        args.historical_training / "best_adapter", model_max_length=2048,
        padding_side="right", local_files_only=True,
    )
    if tokenizer.get_vocab() != old_tokenizer.get_vocab() or old_tokenizer.pad_token_id != 0:
        raise ValueError("fresh and historical vocabularies mismatch")
    def vocab_sha(vocab):
        return digest_bytes(json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode())

    output = {
        "scope": "Actual CPU replay with installed legacy data loader and historical archive",
        "limitations": [
            "Template input agreement covers only archived input and label hash windows.",
            "No LLaMA weights, optimizer tensors, or CUDA RNG are read by this replay.",
            "Training proceeds on one GPU; Torch dropout and reduction numerics can differ.",
        ],
        "inputs": {key: str(value) for key, value in vars(args).items()},
        "seed": cfg.seed, "train_rows": len(train), "validation_rows": len(valid),
        "vocab_sha256": vocab_sha(tokenizer.get_vocab()),
        "sid_vocab_sha256": vocab_sha({token: tokenizer.convert_tokens_to_ids(token) for token in tokens}),
        "fresh_vocab_equals_historical": True,
        "sampler": [],
        "template_hash_checks": [],
    }
    if len(train) != 131413 or len(valid) != 44726 or len(all_prompt["seqrec"]) != 17:
        raise ValueError("historical dataset or template count mismatch")
    collate = Collator(cfg, tokenizer)
    folder = args.historical_training
    generator = torch.Generator()
    output["sampler_initial_seed"] = generator.initial_seed()
    saved_rows = {}
    for epoch in range(1, 5):
        source = folder / f"epoch{epoch}_rows.json"
        archive = json.loads(source.read_bytes())
        rows = sampler_replay(torch, BatchSamplerShard, len(train), generator, sampling)
        if rows != archive["ranks"] or [rows_sha(rank) for rank in rows] != archive["audit"]["rank_order_sha256"]:
            raise ValueError(f"historical sampler showed drift in epoch {epoch}")
        if len(set(rows[0] + rows[1])) != 131413 or sum(map(len, rows)) != 131424:
            raise ValueError(f"historical sampler changed repeat policy in epoch {epoch}")
        saved_rows[epoch] = rows
        output["sampler"].append({
            "epoch": epoch, "archive_sha256": digest_bytes(source.read_bytes()),
            "positions_by_rank": [len(rank) for rank in rows],
            "rank_order_sha256": archive["audit"]["rank_order_sha256"],
            "repeat_row_indices": archive["audit"]["duplicates"],
        })

    for rank in range(2):
        random.seed(cfg.seed)
        source = folder / f"events.rank{rank}.jsonl"
        event_bytes = source.read_bytes()
        events = {}
        for line in event_bytes.splitlines():
            event = json.loads(line)
            if event["event"] == "batch":
                key = (event["epoch_index"] + 1, event["loader_yielded"])
                if key in events:
                    raise ValueError(f"duplicate logged batch {key} on rank {rank}")
                events[key] = event
        checks = []
        for epoch, values in saved_rows.items():
            rows = values[rank]
            for offset in range(0, len(rows), 8):
                key = (epoch, offset // 8 + 1)
                if key not in events:
                    for _ in range(8):
                        random.randint(0, len(all_prompt["seqrec"]) - 1)
                    continue
                event = events[key]
                batch = collate([train[row] for row in rows[offset:offset + 8]])
                input_sha = digest_bytes(batch["input_ids"].numpy().tobytes())
                label_sha = digest_bytes(batch["labels"].numpy().tobytes())
                if (event["input_sha256"] != input_sha or event["label_sha256"] != label_sha
                        or event["shape"] != list(batch["input_ids"].shape)):
                    raise ValueError(f"historical template/input drift on rank {rank}, batch {key}")
                checks.append({"epoch": epoch, "batch": key[1], "shape": event["shape"],
                               "input_sha256": input_sha, "label_sha256": label_sha})
        if not events or len(checks) != len(events):
            raise ValueError(f"missing archived batch coverage on rank {rank}")
        output["template_hash_checks"].append({
            "rank": rank, "archive_sha256": digest_bytes(event_bytes),
            "checks": checks,
        })

    # Native no-split sequential evaluation shards show exactly which repeats
    # participate in the final rank-0 token mean before metric-gather trimming.
    batches = sampling.BatchSampler(range(len(valid)), batch_size=8, drop_last=False)
    eval_tails = [
        list(BatchSamplerShard(batches, num_processes=2, process_index=rank,
                               split_batches=False, even_batches=True))[-1]
        for rank in range(2)
    ]
    if eval_tails != [list(range(len(valid) - 6, len(valid))) + [0, 1], list(range(2, 10))]:
        raise ValueError("historical validation last batch identity")
    output["validation_tail"] = {
        "rows_per_rank": eval_tails, "gather_keeps": 6, "gather_drops": 10,
    }
    output["status"] = "PASS"
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
