"""Bind new GRID SIDs and use C's Normal recipe with evaluation-only V2."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from shuffle import (LEVELS, expected_tokens, fixtures as shuffle_fixtures, shifted,
                     source_positions, verify_input)

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "letter_instruments_grid_normal"
if not BASELINE.is_dir():
    BASELINE = Path(os.environ["GRID_FROZEN_DOWNSTREAM"])
sys.path.extend((str(BASELINE), str(BASELINE / "native")))
import run_normal as original
from candidates import rows, score_words, scores
from data_assets import PREPARED, SPLITS, load, records, sha, split_seq, write
from evaluate import (FP32_TOLERANCE, coverage, fixtures as metric_fixtures,
                      native, stable_order, unified)

TOTAL_USERS = 24772
TOTAL_ITEMS = 9922
PREVIEW_UPDATES = 3
PREVIEW_VAL_BATCHES = 3
PREVIEW_TEST_USERS = 20


def sequences(path):
    raw = load(path)
    assert set(raw) == {str(uid) for uid in range(TOTAL_USERS)}
    series = [raw[str(uid)] for uid in range(TOTAL_USERS)]
    counts = Counter(item for seq in series for item in seq)
    assert set(counts) == set(range(TOTAL_ITEMS)) and min(counts.values()) == 5
    assert counts.total() == 206153 and min(map(len, series)) == 5
    assert all(len(seq) == len(set(seq)) for seq in series)
    return series


def user_axis(path):
    raw = load(path)
    if "user_to_global_row_idx" in raw:
        raw = raw["user_to_global_row_idx"]
    result = {int(uid): idx for uid, idx in raw.items()}
    assert set(result) == set(result.values()) == set(range(TOTAL_USERS))
    assert all(type(idx) is int for idx in result.values())
    return result


def sid_codes(path):
    import torch
    sid = torch.load(path, map_location="cpu", weights_only=True)
    if sid.shape != (4, TOTAL_ITEMS) or sid.dtype != torch.int64:
        raise ValueError(f"Invalid SID axis or dtype: {sid.shape}, {sid.dtype}")
    if not (int(sid.min()) >= 0 and int(sid.max()) < 256):
        raise ValueError("Code outside the native vocabulary")
    codes = list(map(tuple, sid.T.tolist()))
    if len(set(codes)) != TOTAL_ITEMS:
        raise ValueError("Complete SIDs must be unique")
    counts = Counter(code[:3] for code in codes)
    seen = Counter()
    for code in codes:
        prefix = code[:3]
        seen[prefix] += 1
        if code[3] != (0 if counts[prefix] == 1 else seen[prefix]):
            raise ValueError("Fourth code differs from native ordered disambiguation")
    return codes


def select_preview(series, codes):
    shortest = min(range(len(series)), key=lambda uid: len(series[uid]))
    truncated = next(uid for uid, seq in enumerate(series) if len(seq) > 21)
    equal_codes = next(uid for uid, seq in enumerate(series)
                       if any(len(set(codes[item])) < 4 for item in seq[-21:-1]))
    selected = {shortest, truncated, equal_codes, len(series) - 1}
    for uid in range(len(series)):
        if len(selected) == PREVIEW_TEST_USERS:
            break
        selected.add(uid)
    return sorted(selected), {"shortest": shortest, "truncated": truncated, "equal_bare_codes": equal_codes}


def verify_split(path, series, split, uids):
    got = [(row["user_id"], row["sequence_data"]) for row, _ in records(path)]
    expected = [([uid], split_seq(series[uid], split)) for uid in uids]
    assert got == expected, f"Split readback mismatch: {path}"
    return {"rows_read_exact": len(got), "sha256": sha(path),
            "minimum_items": min(len(row) for _, row in got),
            "maximum_items": max(len(row) for _, row in got)}


def prepare(args):
    import tensorflow as tf
    from shutil import copyfile

    tf.config.set_visible_devices([], "GPU")
    root = args.bound / PREPARED
    root.mkdir(parents=True, exist_ok=False)
    source = args.sid / "semantic_ids.pt"
    codes = sid_codes(source)
    dst = root / "semantic_ids.pt"
    copyfile(source, dst)
    assert sha(dst) == sha(source) and sid_codes(dst) == codes
    series = sequences(args.inputs / "Instruments.inter.json")
    row_map = user_axis(args.inputs / "user_axis.json")
    selection, examples = select_preview(series, codes) if args.mode == "preview" else ([], {})
    builds = {f"data/{split}": (split, list(range(TOTAL_USERS))) for split in SPLITS}
    if selection:
        builds["preflight_testing"] = ("testing", selection)
    report = {}
    for relative, (split, uids) in builds.items():
        folder = root / relative
        folder.mkdir(parents=True)
        path = folder / "part-00000.tfrecord.gz"
        with tf.io.TFRecordWriter(str(path), options="GZIP") as stream:
            for uid in uids:
                feature = {name: tf.train.Feature(int64_list=tf.train.Int64List(value=data)) for name, data in
                           (("user_id", [uid]), ("sequence_data", split_seq(series[uid], split)))}
                stream.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
        report[relative] = verify_split(path, series, split, uids)
        # Exercise native TensorFlow decoding, including frame CRC, independently of Python wire parser.
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        data = tf.data.TFRecordDataset(str(path), compression_type="GZIP").with_options(options)
        decoded = []
        for raw in data.as_numpy_iterator():
            f = tf.train.Example.FromString(raw).features.feature
            decoded.append((list(f["user_id"].int64_list.value), list(f["sequence_data"].int64_list.value)))
        assert decoded == [([uid], split_seq(series[uid], split)) for uid in uids]
    catalog = [{"item": item, "sid": sid} for item, sid in enumerate(codes)]
    write(root / "catalog.json", catalog)
    assert [tuple(row["sid"]) for row in load(root / "catalog.json")] == codes
    write(args.bound / "binding.json", {
        "mode": args.mode, "sid_sha256": sha(dst), "gate_sha256": sha(args.sid / "sid_gate.json"),
        "identity_sha256": sha(args.inputs / "item_identity.jsonl"),
        "interactions_sha256": sha(args.inputs / "Instruments.inter.json"),
        "user_axis_sha256": sha(args.inputs / "user_axis.json"),
        "catalog_sha256": sha(root / "catalog.json"), "items": len(codes),
        "sequences_sha256": hashlib.sha256(json.dumps(series, separators=(",", ":")).encode()).hexdigest(),
        "preview_users": selection, "preview_examples": examples, "native_readback": report})
    from shuffle import cpu_replay
    write(args.bound / "full_cpu_replay.json", {"fixtures": shuffle_fixtures(),
                                              "actual_catalog": True, "replay": cpu_replay(series, row_map, codes)})


def check_binding(args):
    root = args.bound / PREPARED
    binding = load(args.bound / "binding.json")
    assert binding["mode"] == args.mode
    for key, artifact in {
        "sid_sha256": root / "semantic_ids.pt", "gate_sha256": args.sid / "sid_gate.json",
        "interactions_sha256": args.inputs / "Instruments.inter.json",
        "user_axis_sha256": args.inputs / "user_axis.json", "identity_sha256": args.inputs / "item_identity.jsonl",
        "catalog_sha256": root / "catalog.json",
    }.items():
        assert sha(artifact) == binding[key], artifact
    assert sha(args.sid / "semantic_ids.pt") == binding["sid_sha256"]
    assert set(binding["native_readback"]) == {f"data/{split}" for split in SPLITS} | (
        {"preflight_testing"} if args.mode == "preview" else set())
    for folder, record in binding["native_readback"].items():
        assert sha(root / folder / "part-00000.tfrecord.gz") == record["sha256"]
    assert (args.bound / "full_cpu_replay.json").is_file()
    return binding


def config(args, destination):
    from omegaconf import OmegaConf, open_dict
    cfg, overrides = original.configure(args.bound, destination, preflight=False)
    if args.mode == "preview":
        params = {"max_steps": PREVIEW_UPDATES, "max_epochs": 1,
                  "val_check_interval": PREVIEW_UPDATES * cfg.trainer.accumulate_grad_batches,
                  "limit_train_batches": PREVIEW_UPDATES * cfg.trainer.accumulate_grad_batches,
                  "limit_val_batches": PREVIEW_VAL_BATCHES}
        with open_dict(cfg.trainer):
            for key, value in params.items():
                cfg.trainer[key] = value
        overrides.extend(f"preview: trainer.{key}={value}" for key, value in params.items())
        folder = str(args.bound / PREPARED / "preflight_testing")
        cfg.data_loading.test_dataloader_config.dataloader.data_folder = folder
        overrides.append(f"preview: native test folder={folder}")
    assert cfg.ckpt_path is None and not cfg.model.shuffle_embedding_on_test
    OmegaConf.save(cfg, destination / "effective_config.yaml", resolve=True)
    write(destination / "overrides.json", overrides)
    return cfg


def fit(args):
    import lightning as L
    from lightning.pytorch.callbacks import Callback
    import torch
    from src.utils.launcher_utils import pipeline_launcher

    binding = check_binding(args)
    output = args.output
    output.mkdir(parents=True)
    cfg = config(args, output)
    codes = sid_codes(args.bound / PREPARED / "semantic_ids.pt")
    expanded = Counter()
    updates, validations = [], []
    runtime_start = time.monotonic()

    class TrainEvidence(Callback):
        def on_train_start(self, trainer, model):
            assert trainer.global_step == 0 and not trainer.optimizers[0].state
            self.last_validation = time.monotonic()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            assert not model.shuffle_embedding_on_test and model.shuffle_calls == 0
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            assert math.isfinite(float(loss.detach()))
            (pair,) = batch
            batch_size, width = pair[0].mask.shape
            assert width == 120 and 0 < batch_size <= 128
            expanded[batch_size] += 1
            if args.mode == "preview" and updates and "after" not in updates[-1]:
                if trainer.global_step == updates[-1]["step_before"] + 1:
                    self.update_result(trainer, model)

        def on_before_optimizer_step(self, trainer, model, optimizer):
            if args.mode == "preview":
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(grad).all().item() for grad in grads)
                param = model.item_sid_embedding_table_encoder.weight
                assert float(param.grad.norm()) > 0
                previous = original.tensor_digest(param)
                updates.append({"before": previous, "gradient_l2": float(param.grad.norm()),
                                "step_before": trainer.global_step})

        def update_result(self, trainer, model):
            param = model.item_sid_embedding_table_encoder.weight
            observed = original.tensor_digest(param)
            assert updates[-1]["before"] != observed
            adam_step = int(trainer.optimizers[0].state[param]["step"])
            assert adam_step == updates[-1]["step_before"] + 1
            updates[-1].update(after=observed, adam_step=adam_step)

        def on_validation_start(self, trainer, model):
            assert model.shuffle_calls == 0 and not model.shuffle_embedding_on_test
            self.val_start = time.monotonic()
            self.val_batches = 0

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            self.val_batches += 1

        def on_validation_end(self, trainer, model):
            metric = float(trainer.callback_metrics["val/recall@5"])
            assert math.isfinite(metric)
            validations.append({"step": trainer.global_step, "batches": self.val_batches,
                                "recall_at_5": metric, "seconds": time.monotonic() - self.val_start,
                                "train_seconds_since_validation": self.val_start - self.last_validation})
            self.last_validation = time.monotonic()
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(validations[-1], allow_nan=False) + "\n")

    recorder = TrainEvidence()
    with pipeline_launcher(cfg) as run:
        model, trainer = run.model, run.trainer
        write(output / "model.json", original.model_identity(model, codes))
        trainer.callbacks.append(recorder)
        L.seed_everything(original.SEED, workers=True)
        trainer.fit(model, datamodule=run.datamodule, ckpt_path=None)
        if args.mode == "preview":
            if "after" not in updates[-1]:
                recorder.update_result(trainer, model)
            assert trainer.global_step == len(updates) == PREVIEW_UPDATES
            assert sum(expanded.values()) == cfg.trainer.limit_train_batches
            assert len(validations) == 1 and validations[0]["batches"] == PREVIEW_VAL_BATCHES
        best = Path(trainer.checkpoint_callback.best_model_path).resolve()
        assert best.is_relative_to(output.resolve()) and trainer.checkpoint_callback.monitor == "val/recall@5"
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        assert checkpoint["optimizer_states"][0]["state"] and set(checkpoint["state_dict"]) == set(model.state_dict())
        checkpoint_score = float(trainer.checkpoint_callback.best_model_score)
        assert checkpoint_score == max(row["recall_at_5"] for row in validations)
        write(output / "best.json", {"file": str(best.relative_to(output)), "sha256": sha(best),
                                    "state_sha256": original.state_digest(checkpoint["state_dict"]),
                                    "mode": args.mode, "binding_sha256": sha(args.bound / "binding.json"),
                                    "sid_sha256": binding["sid_sha256"], "best_step": checkpoint["global_step"],
                                    "score": checkpoint_score})
        write(output / "training.json", {"seconds": time.monotonic() - runtime_start,
                                         "updates": trainer.global_step, "expanded_microbatch": dict(expanded),
                                         "optimizer_evidence": updates, "validation": validations,
                                         "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                                         "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                                         "train_seed": original.SEED, "legacy_shuffle_calls": model.shuffle_calls})


class EncoderV2:
    """Scope the shared-table prehook to this model's encoder call."""
    def __init__(self, model, series, codes, row_map, condition, detailed):
        self.model = model
        self.series = series
        self.codes = codes
        self.row_map = row_map
        self.condition = condition
        self.detailed = detailed
        self.calls = []

    def forward(self, attention_mask, input_ids, user_id, output_attentions=False):
        import torch
        m = self.model
        assert m.is_testing and not m.training and not m.shuffle_embedding_on_test
        assert not output_attentions and m.num_hierarchies == 4 and m.num_embeddings_per_hierarchy == 256
        user = user_id[:, 0].detach().cpu().tolist()
        inputs = input_ids.detach().cpu().tolist()
        masks = attention_mask.detach().cpu().tolist()
        before, after, orders = [], [], []
        for uid, values, mask in zip(user, inputs, masks, strict=True):
            history = self.series[uid][-21:-1]
            verify_input(values, mask, history, self.codes)
            raw = shifted(values, mask)
            plan = list(source_positions(self.condition, self.row_map[uid], len(history)))
            before.append(raw)
            after.append(expected_tokens(raw, len(history), plan))
            orders.append(plan)
        table = m.get_embedding_table("encoder")
        visited = []
        embeddings = []

        def adapt(module, arguments):
            (native_tokens,) = arguments
            assert native_tokens.detach().cpu().tolist() == before
            visited.append(1)
            if before != after:
                return (native_tokens.new_tensor(after),)
            return None

        def observe_table(module, arguments, output):
            (tokens,) = arguments
            assert tokens.detach().cpu().tolist() == after
            embeddings.append(output)

        def observe_encoder(module, args, kwargs):
            values, mask = kwargs["sequence_embedding"], kwargs["attention_mask"]
            assert len(embeddings) == 1
            assert torch.equal(values.reshape(len(user), 30, 5, -1)[:, :, :4].reshape_as(embeddings[0]),
                               embeddings[0])
            assert torch.equal(values[:, 4::5], m.sep_token.expand_as(values[:, 4::5]))
            reshaped = attention_mask.reshape(len(user), 30, 4)
            assert torch.equal(mask, torch.cat((reshaped, reshaped[:, :, -1:]), dim=2).reshape(len(user), 150))

        with ExitStack() as hooks:
            hooks.enter_context(table.register_forward_pre_hook(adapt))
            if self.detailed:
                hooks.enter_context(table.register_forward_hook(observe_table))
                hooks.enter_context(m.encoder.register_forward_pre_hook(observe_encoder, with_kwargs=True))
            result = self.native_forward(attention_mask=attention_mask, input_ids=input_ids,
                                         user_id=user_id, output_attentions=output_attentions)
        assert visited == [1]
        self.calls.append((user, before, after, orders))
        return result

    def take(self, users):
        assert len(self.calls) == 2 and self.calls[0] == self.calls[1], "Loss/generate encoder interventions differ"
        result = self.calls[0]
        self.calls.clear()
        assert result[0] == users
        return result[1:]


def evaluate_condition(args):
    import lightning as L
    import torch
    from src.utils.launcher_utils import pipeline_launcher

    binding = check_binding(args)
    best = load(args.training / "best.json")
    assert best["mode"] == args.mode and best["sid_sha256"] == binding["sid_sha256"]
    assert best["binding_sha256"] == sha(args.bound / "binding.json")
    path = (args.training / best["file"]).resolve()
    assert path.is_relative_to(args.training.resolve()) and sha(path) == best["sha256"]
    expected = binding["preview_users"] if args.mode == "preview" else list(range(TOTAL_USERS))
    series = sequences(args.inputs / "Instruments.inter.json")
    codes = sid_codes(args.bound / PREPARED / "semantic_ids.pt")
    row_map = user_axis(args.inputs / "user_axis.json")
    output = args.output
    output.mkdir(parents=True)
    cfg = config(args, output)
    condition = args.condition

    def testing(run, name, v2):
        model, trainer = run.model, run.trainer
        original.model_identity(model, codes)
        model.strict_loading = True
        saved_load = model.load_state_dict
        loads = []
        ranks = {}
        checks = Counter()
        adapter = EncoderV2(model, series, codes, row_map, condition, detailed=args.mode == "preview")
        adapter.native_forward = model.encoder_forward_pass

        def load_state(values, strict=True, **kwargs):
            assert strict is True and original.state_digest(values) == best["state_sha256"]
            result = saved_load(values, strict=True, **kwargs)
            assert not result.missing_keys and not result.unexpected_keys
            assert original.state_digest(model.state_dict()) == best["state_sha256"]
            loads.append({"strict": True, "keys": len(values), "checkpoint_sha256": best["sha256"],
                          "state_sha256": best["state_sha256"]})
            return result

        def collect_rank(key, value):
            assert key not in ranks
            ranks[key] = value.detach().cpu().tolist()

        seen = []
        started = time.monotonic()
        filename = output / name
        with ExitStack() as stack:
            stream = stack.enter_context(filename.open("x", encoding="utf-8"))

            def capture(inputs, labels, ids, probabilities):
                user_ids = inputs.transformed_sequences["user_id"][:, 0].detach().cpu().tolist()
                values = inputs.transformed_sequences["sequence_data"].detach().cpu().tolist()
                masks = inputs.mask.detach().cpu().tolist()
                target = labels.labels["sequence_data"].reshape(-1, 4).detach().cpu().tolist()
                assert probabilities.dtype == torch.float32
                assert list(ids.shape) == [len(user_ids), 10, 4]
                assert list(probabilities.shape) == [len(user_ids), 10]
                assert set(ranks) == {"recall@5", "recall@10", "ndcg@5", "ndcg@10"}
                histories = [series[uid][-21:-1] for uid in user_ids]
                if v2:
                    before, after, orders = adapter.take(user_ids)
                else:
                    before = after = [shifted(v, mask) for v, mask in zip(values, masks)]
                    orders = [list(source_positions("Normal", row_map[uid], len(hist)))
                              for uid, hist in zip(user_ids, histories)]
                for i, (uid, sid_list, score_list) in enumerate(zip(user_ids, ids.cpu().tolist(), probabilities.cpu().tolist())):
                    hist, target_item = histories[i], series[uid][-1]
                    verify_input(values[i], masks[i], hist, codes)
                    assert target[i] == list(codes[target_item])
                    checks["history_groups"] += len(hist)
                    checks["nonidentity_draws"] += sum(index != 0 for _, index, _ in orders[i])
                    checks["changed_token_groups"] += sum(
                        before[i][j:j+4] != after[i][j:j+4] for j in range(0, 4 * len(hist), 4))
                    record = {"user_id": str(uid), "global_row": row_map[uid], "mode": condition,
                              "target_item_id": target_item, "target_sid": target[i],
                              "history_item_ids": hist, "input_ids": values[i], "input_mask": masks[i],
                              "offset_ids_before": before[i], "offset_ids_after": after[i],
                              "draws": orders[i], "ids": sid_list, "scores_f32le": score_words(score_list),
                              "native_topk": {key: index[i] for key, index in ranks.items()}}
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                seen.extend(user_ids)
                checks[f"batch_size_{len(user_ids)}"] += 1
                ranks.clear()

            stack.enter_context(patch.object(model, "normal_candidate_sink", capture, create=True))
            for key, metric in model.evaluator.metrics.items():
                stack.enter_context(patch.object(metric, "normal_rank_sink",
                                                lambda value, key=key: collect_rank(key, value), create=True))
            if v2:
                stack.enter_context(patch.object(model, "encoder_forward_pass", adapter.forward))
            stack.enter_context(patch.object(model, "load_state_dict", load_state))
            L.seed_everything(original.EVAL_SEED, workers=True)
            metrics, = trainer.test(model, datamodule=run.datamodule, ckpt_path=str(path))
        assert len(loads) == 1 and not model.shuffle_calls and not adapter.calls
        assert all(math.isfinite(v) for v in metrics.values())
        coverage(seen, expected)
        assert checks["batch_size_8"] == len(expected) // 8 and checks["batch_size_4"] == 1
        return {"metrics": metrics, "seconds": time.monotonic() - started, "checks": dict(checks),
                "reload": loads[0], "candidate_sha256": sha(filename),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}

    with pipeline_launcher(cfg) as run:
        receipt = testing(run, "candidates.jsonl", v2=condition != "Normal" or args.mode == "preview")
    if condition == "Normal" and args.mode == "preview":
        with pipeline_launcher(cfg) as run:
            shadow = testing(run, "native_shadow.jsonl", v2=False)
        assert shadow["metrics"] == receipt["metrics"]
        assert shadow["candidate_sha256"] == receipt["candidate_sha256"], "Native Normal and identity path differ"
        receipt["native_shadow"] = shadow
    receipt["eval_seed"] = original.EVAL_SEED
    receipt["sid_sha256"] = binding["sid_sha256"]
    write(output / "native_evaluation.json", receipt)
    with (output / "per_user_metrics.jsonl").open("x", encoding="utf-8") as stream:
        result = replay_candidates(output / "candidates.jsonl", condition, series, codes, row_map,
                                   expected, receipt["metrics"], stream)
    write(output / "offline_evaluation.json", result)


def replay_candidates(path, condition, series, codes, row_map, uids, native_metrics, stream=None):
    sizes = Counter(codes)
    result_rows = list(rows(path))
    coverage([int(row["user_id"]) for row in result_rows], uids)
    populations = {"native": [], "unified": []}
    checks = Counter()
    for record, uid in zip(result_rows, uids, strict=True):
        hist = series[uid][-21:-1]
        target = codes[series[uid][-1]]
        assert record["mode"] == condition and record["global_row"] == row_map[uid]
        assert record["target_item_id"] == series[uid][-1] and record["target_sid"] == list(target)
        assert record["history_item_ids"] == hist
        verify_input(record["input_ids"], record["input_mask"], hist, codes)
        before = shifted(record["input_ids"], record["input_mask"])
        plans = list(source_positions(condition, row_map[uid], len(hist)))
        assert record["draws"] == json.loads(json.dumps(plans))
        assert record["offset_ids_before"] == before
        assert record["offset_ids_after"] == expected_tokens(before, len(hist), plans)
        values = scores(record)
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        sids = [tuple(sid) for sid in record["ids"]]
        assert len(sids) == 10 and all(len(sid) == 4 and all(type(v) is int and 0 <= v < 256 for v in sid) for sid in sids)
        order = stable_order(values)
        nat = native(record, values)
        uni, first = unified([sids[i] for i in order], target, sizes)
        populations["native"].append(nat)
        populations["unified"].append(uni)
        if stream is not None:
            stream.write(json.dumps({"global_row": row_map[uid], "user_id": str(uid), "unified_order": order,
                                     "first_sid_rank": first, "native": nat, **uni}, allow_nan=False) + "\n")
        checks.update(candidates=10, out_of_catalog=sum(sid not in sizes for sid in sids),
                      duplicate_sids=10 - len(set(sids)), tied_score_rows=len(set(values)) != 10,
                      native_stable_order_differences=any(v != order[:len(v)] for v in record["native_topk"].values()),
                      collision_target_rows=sizes[target] > 1)
    averages = {group: {key: math.fsum(row[key] for row in population) / len(uids)
                        for key in population[0]} for group, population in populations.items()}
    differences = {key: abs(val - native_metrics["test/" + key]) for key, val in averages["native"].items()}
    assert max(differences.values()) <= FP32_TOLERANCE
    assert checks["candidates"] == 10 * len(uids)
    return {"coverage": len(uids), "counts": dict(checks), "metrics": averages, "native_abs_delta": differences}


def summarize(args):
    directory = args.output
    ledgers = {mode: load(directory / mode / "offline_evaluation.json") for mode in LEVELS}
    native_records = {mode: load(directory / mode / "native_evaluation.json") for mode in LEVELS}
    ckpts = {data["reload"]["checkpoint_sha256"] for data in native_records.values()}
    assert len(ckpts) == 1
    baselines = list(rows(directory / "Normal" / "candidates.jsonl"))
    delta = {}
    for mode in ("Shuffle3a", "Shuffle4"):
        summary, base = ledgers[mode]["metrics"], ledgers["Normal"]["metrics"]
        delta[mode] = {"metrics": {key: {name: {"absolute": value - base[key][name],
                                               "relative": (value - base[key][name]) / base[key][name] if base[key][name] else None}
                                        for name, value in group.items()} for key, group in summary.items()}}
        changed_ids = changed_scores = 0
        for normal, perturbed in zip(baselines, rows(directory / mode / "candidates.jsonl"), strict=True):
            for name in ("user_id", "global_row", "target_item_id", "target_sid", "history_item_ids",
                         "input_ids", "input_mask", "offset_ids_before"):
                assert normal[name] == perturbed[name]
            changed_ids += normal["ids"] != perturbed["ids"]
            changed_scores += normal["scores_f32le"] != perturbed["scores_f32le"]
        delta[mode].update(candidate_id_rows_changed=changed_ids, candidate_score_rows_changed=changed_scores,
                           **native_records[mode]["checks"])
    write(directory / "summary.json", {"mode": args.mode, "best_sha256": ckpts.pop(),
                                       "candidate_metrics": ledgers, "relative_to_normal": delta,
                                       "metric_fixtures": metric_fixtures()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("plan", "prepare", "fit", "evaluate", "summarize"))
    p.add_argument("--mode", choices=("preview", "formal"), required=True)
    for name in ("inputs", "sid", "bound", "output", "training"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--condition", choices=LEVELS)
    args = p.parse_args()
    os.environ.update(OMP_NUM_THREADS="4", TF_CPP_MIN_LOG_LEVEL="3", TF_NUM_INTRAOP_THREADS="2",
                      TF_NUM_INTEROP_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8", HYDRA_FULL_ERROR="1",
                      PROJECT_ROOT=str(original.NATIVE_ROOT))
    if args.stage in ("fit", "evaluate"):
        import torch
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("medium")
        assert torch.cuda.device_count() == 1
        assert original.verify_sources()
    if args.stage == "plan":
        args.output.mkdir(parents=True)
        config(args, args.output)
        return
    {"prepare": prepare, "fit": fit, "evaluate": evaluate_condition, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
