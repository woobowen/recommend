"""Recompute the unchanged original metrics from complete saved predictions."""
import collections
import json
from pathlib import Path

from .common import sha256, write_json


def recompute(spec, cfg, predictions, output, coverage_index=None, trainer_metrics=None):
    import hydra
    import torch
    evaluator = hydra.utils.instantiate(cfg.eval.evaluator)
    sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True)
    legal = set(map(tuple, sid.t().tolist()))
    counts = collections.Counter()
    invalid, duplicates, rows = 0, 0, 0
    per_sample = output / "per_sample_metrics.jsonl"
    with Path(predictions).open() as stream, per_sample.open("w") as per:
        pending = []
        def process(batch):
            nonlocal invalid, duplicates, rows
            ids = torch.tensor([r["sids"] for r in batch], dtype=torch.long)
            scores = torch.tensor([r["scores"] for r in batch], dtype=torch.float32)
            labels = torch.tensor([r["target"] for r in batch], dtype=torch.long)
            assert ids.shape[1:] == (spec["generation_k"], 4) and torch.isfinite(scores).all()
            evaluator(marginal_probs=scores, generated_ids=ids, labels=labels)
            target = (ids == labels[:,None,:]).all(-1).int()
            each = {name: metric._metric(scores, target).tolist() for name,metric in evaluator.metrics.items()}
            for i,r in enumerate(batch):
                tuples = list(map(tuple,r["sids"]))
                invalid += sum(v not in legal for v in tuples)
                duplicates += len(tuples) - len(set(tuples))
                counts[(r["sample_key"], tuple(r["target"]))] += 1
                per.write(json.dumps({"sample_key": r["sample_key"], "metrics": {k:v[i] for k,v in each.items()}}) + "\n")
                rows += 1
        for line in stream:
            pending.append(json.loads(line))
            if len(pending) == 8:
                process(pending); pending=[]
        if pending:
            process(pending)
    values = {name:float(metric.compute()) for name,metric in evaluator.metrics.items()}
    expected = None
    if coverage_index is not None:
        expected = collections.Counter()
        for line in Path(coverage_index).read_text().splitlines():
            row=json.loads(line);expected[(row["sample_key"],tuple(row["target"]))]+=1
        assert counts == expected, "Saved predictions do not cover the exact original target multiset"
    differences = {}
    if trainer_metrics is not None:
        reference=json.loads(Path(trainer_metrics).read_text())["metrics"][0]
        differences={name:abs(values[name]-float(reference["test/"+name])) for name in values}
        assert all(v <= 1e-7 for v in differences.values()), "Independent metric aggregation differs"
    result={"status":"PASS", "metrics":values,"samples":rows,"unique_users":len({k[0] for k in counts}),
        "prediction_sha256":sha256(predictions),"invalid_candidate_count":invalid,"duplicate_candidate_count":duplicates,
        "candidate_treatment":"original evaluator, no filtering, completion, rescoring, or reordering",
        "coverage_exact":expected is not None,"trainer_absolute_differences":differences,
        "per_sample_metrics_sha256":sha256(per_sample)}
    write_json(output / "recomputed_metrics.json",result)
    return result
