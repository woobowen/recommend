"""Read-only observation of native test tensors and the evaluator's actual top-k indices."""
import json
from pathlib import Path
import struct
import time

def score_words(scores):
    return struct.pack("<10f", *scores).hex()


class CandidateSink:
    def __init__(self, output, model, axes, active=True):
        self.stream = output.open("x", encoding="utf-8") if active else None
        self.model = model
        self.axes = axes
        self.ranks = {}
        self.users = []
        self.generation_calls = 0
        self.write_seconds = 0.0

    def __enter__(self):
        if self.stream is not None:
            self.model.normal_candidate_sink = self.capture
            for name, metric in self.model.evaluator.metrics.items():
                metric.normal_rank_sink = lambda value, name=name: self.rank(name, value)
        return self

    def __exit__(self, *exc):
        if self.stream is not None:
            del self.model.normal_candidate_sink
            for metric in self.model.evaluator.metrics.values():
                del metric.normal_rank_sink
            self.stream.close()

    def rank(self, name, value):
        assert name not in self.ranks
        self.ranks[name] = value.detach().cpu().tolist()

    def capture(self, inputs, labels, ids, scores):
        assert self.model.is_testing and not self.model.training and self.model.shuffle_calls == 0
        user_ids = inputs.transformed_sequences["user_id"][:, 0].detach().cpu().tolist()
        sequences = inputs.transformed_sequences["sequence_data"].detach().cpu().tolist()
        masks = inputs.mask.detach().cpu().tolist()
        target = labels.labels["sequence_data"].reshape(-1, 4).detach().cpu().tolist()
        assert tuple(ids.shape) == (len(user_ids), 10, 4)
        assert tuple(scores.shape) == (len(user_ids), 10) and str(scores.dtype) == "torch.float32"
        candidates, probabilities = ids.detach().cpu().tolist(), scores.detach().cpu().tolist()
        assert set(self.ranks) == {"recall@5", "recall@10", "ndcg@5", "ndcg@10"}
        started = time.monotonic()
        for i, uid in enumerate(user_ids):
            sequence = self.axes[uid]
            record = {
                "global_row": uid, "user_id": str(uid), "target_item_id": sequence[-1],
                "target_sid": target[i], "history_item_ids": sequence[:-1],
                "input_ids": sequences[i], "input_mask": masks[i],
                "ids": candidates[i], "scores_f32le": score_words(probabilities[i]),
                "native_topk": {name: rank[i] for name, rank in self.ranks.items()},
            }
            self.stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.stream.flush()
        self.write_seconds += time.monotonic() - started
        self.ranks.clear()
        self.users.extend(user_ids)
        self.generation_calls += 1


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def scores(record):
    return struct.unpack("<10f", bytes.fromhex(record["scores_f32le"]))
