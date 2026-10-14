"""Saved Beauty execution entrypoint. Run one route per independent process."""
import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .common import REPO, bind_source, write_json, sha256


def model_arguments(batch):
    x = batch[0]
    return dict(attention_mask=x.mask, input_ids=x.transformed_sequences["sequence_data"],
                user_id=x.transformed_sequences.get("user_id"))


def set_testing(model, value):
    model.is_testing = value


def clear_runtime(model):
    if hasattr(model, "clear_runtime_state"):
        model.clear_runtime_state()
    if hasattr(model, "is_testing"):
        model.is_testing = False


def numeric_difference(a, b):
    import torch
    return {"max_absolute": float((a - b).abs().max()),
            "allclose": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6)),
            "exact": bool(torch.equal(a, b))}


def check_data(spec, cfg, output):
    import hydra
    import torch
    from itertools import islice
    from .common import verify_asset, make_loader, batch_keys
    sid = torch.load(spec["sid"]["path"], weights_only=True, map_location="cpu")
    assert sid.dtype in (torch.int16, torch.int32, torch.int64) and sid.ndim == 2 and sid.shape[0] == 4 and sid.min() >= 0
    first3 = sid[:3].t()
    unique, inverse, counts = first3.unique(dim=0, return_inverse=True, return_counts=True)
    fourth = torch.zeros(sid.shape[1], dtype=torch.long)
    for group in (counts > 1).nonzero().flatten():
        indexes = (inverse == group).nonzero().flatten()
        fourth[indexes] = torch.arange(1, indexes.numel() + 1)
    assert torch.equal(fourth, sid[3].long()), "Fourth-code collision semantics differ from preserved producer"
    checks = []
    for split in ("train", "val", "test"):
        lc = hydra.utils.instantiate(cfg.data_loading[f"{split}_dataloader_config"].dataloader)
        files = sorted(Path(lc.data_folder).glob("*.tfrecord.gz"))
        assert files and all(os.access(p, os.R_OK) for p in files)
        iterator = lc.dataset_config.data_iterator
        iterator.update_list_of_file_paths([str(files[0])]); iterator.should_shuffle_rows = False
        semantic = []
        prepared_rows = []
        target_sids = []
        for raw in islice(iterator.iterrows(), 8):
            ids = torch.as_tensor(raw["sequence_data"].values.numpy()).long()
            assert ids.numel() > 0 and ids.min() >= 0 and ids.max() < sid.shape[1]
            row = raw
            for pre in lc.dataset_config.preprocessing_functions:
                row = pre(row, dataset_config=lc.dataset_config)
                assert row is not None
            mapped = row["sequence_data"]
            assert mapped.dtype in (torch.int16, torch.int32, torch.int64)
            expected = sid.t()[ids].reshape(-1).long()
            assert torch.equal(mapped, expected)
            semantic.append({"raw_items": ids.numel(), "dtype": str(mapped.dtype),
                             "first_item_sids": sid[:, ids[:3]].t().tolist(),
                             "target_sid": sid[:, ids[-1]].tolist()})
            prepared_rows.append(row)
            target_sids.append(sid[:, ids[-1]].long())
        if split != "train":
            real_input, real_labels = lc.collate_fn(prepared_rows, labels=lc.labels,
                sequence_length=lc.sequence_length, masking_token=lc.masking_token,
                padding_token=lc.padding_token, oov_token=lc.get("oov_token", None))
            actual_targets = next(iter(real_labels.labels.values())).reshape(len(prepared_rows), 4)
            assert torch.equal(actual_targets.long(), torch.stack(target_sids)), "Raw item to collated target differs"
            locations = next(iter(real_labels.label_location.values())).long()
            assert not real_input.mask[locations[:,0],locations[:,1]].any(), "Target visible to encoder"
        checks.append({"split": split, "files": len(files), "num_workers": lc.num_workers,
            "drop_last": lc.drop_last, "batch_rows": lc.batch_size_per_device,
            "sequence_length_tokens_including_target": lc.sequence_length,
            "visible_history_item_limit": lc.sequence_length // 4 - 1,
            "semantic_examples": semantic, "assign_all_files_per_worker": getattr(lc, "assign_all_files_per_worker", False)})
    loader = make_loader(cfg, "val", limit_rows=19)
    batches = list(loader)
    details = []
    for batch in batches:
        x, labels = batch
        values = x.transformed_sequences["sequence_data"]
        assert values.shape[1] % 4 == 0
        groups = x.mask.reshape(values.shape[0], -1, 4)
        assert torch.all(groups == groups[..., :1]), "Partial valid item group"
        y = next(iter(labels.labels.values())).reshape(values.shape[0], 4)
        assert not (y < 0).any()
        assert all(tuple(v) in set(map(tuple, sid.t().tolist())) for v in y.tolist())
        details.append({"batch_shape": list(values.shape), "target_shape": list(y.shape),
                        "visible_items": groups.sum(-1).gt(0).sum(-1).tolist(), "keys": batch_keys(batch)})
    write_json(output / "data_check.json", {"sid_shape": list(sid.shape), "sid_dtype": str(sid.dtype),
        "range": [int(sid.min()), int(sid.max())], "fourth_collision_rule_verified": True,
        "split_checks": checks, "bounded_validation": details, "source_data_modified": False})
    return batches


def permutation_check(model, output):
    import torch
    from .baseline import NATIVE_SHUFFLE
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 12, 5)
    mask = torch.ones(2, 12, dtype=torch.long)
    mask[1, 8:] = 0
    checks = []
    for n in (3, 4):
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        gen = torch.Generator().set_state(before)
        perm = torch.rand(2, 3, n, generator=gen).argsort(2)
        actual = NATIVE_SHUFFLE(model, x.clone(), mask, 4, n)
        grouped = x.reshape(2, 3, 4, 5)
        expected = grouped.clone()
        changed = grouped[:, :, :n].gather(2, perm[..., None].expand(-1, -1, -1, 5))
        expected[:, :, :n] = torch.where(mask.reshape(2, 3, 4)[..., :1, None].bool(), changed, expected[:, :, :n])
        assert torch.equal(actual, expected.reshape_as(x))
        assert not torch.equal(actual, x)
        assert torch.equal(actual[1, 8:], x[1, 8:])
        if n == 3:
            assert torch.equal(actual.reshape_as(grouped)[:, :, 3], grouped[:, :, 3])
        checks.append({"n": n, "permutations": perm.tolist(), "exact_expected": True,
                       "padding_unchanged": True, "fourth_unchanged": n == 3})
    write_json(output / "identifiable_permutations.json", checks)


class PredictionRecorder:
    """Wrap the unchanged evaluator; recording neither rescores nor filters."""
    def __init__(self, base, path):
        self.base, self.path = base, Path(path)
        self.current_keys = None
        self.count = 0

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __call__(self, **kwargs):
        self.base(**kwargs)
        ids = kwargs["generated_ids"].detach().cpu()
        scores = kwargs["marginal_probs"].detach().cpu()
        labels = kwargs["labels"].detach().cpu().reshape(ids.shape[0], -1)
        assert self.current_keys is not None and len(self.current_keys) == ids.shape[0]
        with self.path.open("a") as stream:
            for key, pred, score, target in zip(self.current_keys, ids.tolist(), scores.tolist(), labels.tolist()):
                stream.write(json.dumps({"sample_key": key, "sids": pred, "scores": score, "target": target}, allow_nan=False) + "\n")
                self.count += 1


def test_lifecycle(cfg, model, dm, output, batches=None):
    import hydra
    from lightning import Callback
    from torch.utils.data import DataLoader
    from .common import batch_keys
    output.mkdir(parents=True, exist_ok=True)
    original_evaluator = model.evaluator
    recorder = PredictionRecorder(original_evaluator, output / "predictions.jsonl")
    model.evaluator = recorder
    class Keys(Callback):
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            recorder.current_keys = batch_keys(batch)
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if batch_idx % 64 == 0:
                write_json(output / "progress.json", {"stage":"test", "batch":batch_idx,"samples":recorder.count})
        def on_exception(self, trainer, pl_module, exception):
            clear_runtime(pl_module)
    trainer_cfg = cfg.trainer.copy()
    trainer_cfg.pop("profiler", None)
    trainer = hydra.utils.instantiate(trainer_cfg, logger=False, callbacks=[Keys()], enable_checkpointing=False)
    try:
        if batches is None:
            metrics = trainer.test(model, datamodule=dm, ckpt_path=None)
        else:
            metrics = trainer.test(model, dataloaders=DataLoader(batches, batch_size=None), ckpt_path=None)
    finally:
        clear_runtime(model)
        model.evaluator = original_evaluator
    write_json(output / "metrics.json", {"metrics": metrics, "samples": recorder.count,
                                          "formal": batches is None, "checkpoint_load": "strict before Trainer.test"})
    return metrics


def smoke(spec, cfg, model, dm, output, batches):
    import hydra
    import torch
    from .common import batch_to
    device = torch.device("cuda:0")
    model.to(device).eval()
    batch = batch_to(batches[0], device)
    args = model_arguments(batch)
    original_cfg = cfg.model.copy()
    original_name = {"B":"tiger_generation_model","A":"tiger_generation_model_shuffle","MLP":"tiger_generation_model_mlp"}[spec["route"]]
    original_cfg._target_ = "src.models.modules.semantic_id." + original_name + ".SemanticIDEncoderDecoder"
    if spec["route"] == "B":
        original_cfg.pop("shuffle_n", None)
    if spec["route"] == "MLP":
        original_cfg.pop("save_attentions_on_test", None)
    original = hydra.utils.instantiate(original_cfg)
    original.load_state_dict(model.state_dict(), strict=True)
    original.to(device).eval()
    set_testing(model, False)
    rng_before = torch.cuda.get_rng_state().clone()
    cpu_rng_before = torch.get_rng_state().clone()
    with torch.inference_mode():
        logits, loss = model.model_step(batch[0], batch[1])
        reference_logits, reference_loss = original.model_step(batch[0], batch[1])
        ids, scores = model.generate(**args)
        ref_ids, ref_scores = original.generate(**args)
        again, again_scores = model.generate(**args)
        reversed_args = {k: v.flip(0) if v is not None else None for k, v in args.items()}
        rev, rev_scores = model.generate(**reversed_args)
        single_args = {k: v[:1] if v is not None else None for k, v in args.items()}
        single, single_scores = model.generate(**single_args)
    checks = {"decoder_hidden": numeric_difference(logits, reference_logits), "loss": numeric_difference(loss, reference_loss),
        "scores": numeric_difference(scores, ref_scores), "sid_exact": torch.equal(ids, ref_ids),
        "repeat_sid_exact": torch.equal(ids, again), "repeat_scores": numeric_difference(scores, again_scores),
        "sample_order_sid_exact": torch.equal(ids, rev.flip(0)), "sample_order_scores": numeric_difference(scores, rev_scores.flip(0)),
        "batch_size_sid_exact": torch.equal(ids[:1], single), "batch_size_scores": numeric_difference(scores[:1], single_scores),
        "off_rng_unchanged": torch.equal(rng_before, torch.cuda.get_rng_state()),
        "off_cpu_rng_unchanged": torch.equal(cpu_rng_before, torch.get_rng_state()),
        "shared_sid_object": model.get_embedding_table("encoder") is model.get_embedding_table("decoder"),
        "finite_loss": bool(torch.isfinite(loss))}
    with torch.inference_mode():
        checks["head_logits"] = [numeric_difference(model.decoder.decoder_mlp[h](logits[:,h]), original.decoder.decoder_mlp[h](reference_logits[:,h])) for h in range(model.num_hierarchies)]
        original_single, original_single_scores = original.generate(**single_args)
        checks["original_batch_size_scores"] = numeric_difference(ref_scores[:1], original_single_scores)
        checks["adapter_vs_original_single_scores"] = numeric_difference(single_scores, original_single_scores)
        checks["adapter_vs_original_single_sid_exact"] = torch.equal(single, original_single)
        back_ids, back_scores = model.generate(**args)[:2]
        checks["batch_size_roundtrip_sid_exact"] = torch.equal(back_ids, ids)
        checks["batch_size_roundtrip_scores"] = numeric_difference(back_scores, scores)
    # Preserve diagnostics before assertions, including failed attempts.
    write_json(output / "off_and_cache.json", checks)
    assert all(c["allclose"] for c in checks["head_logits"])
    assert all(checks[k]["allclose"] for k in ("decoder_hidden", "loss", "scores", "repeat_scores", "sample_order_scores", "adapter_vs_original_single_scores", "batch_size_roundtrip_scores"))
    assert checks["sid_exact"]
    assert checks["adapter_vs_original_single_sid_exact"] and checks["batch_size_roundtrip_sid_exact"]
    checks["cross_batch_numeric_boundary"] = "Retained original-model shape-dependent drift; fixed-shape adapter/original and batch-size roundtrip required at original tolerances"
    assert checks["repeat_sid_exact"] and checks["sample_order_sid_exact"] and checks["batch_size_sid_exact"]
    assert checks["off_rng_unchanged"] and checks["off_cpu_rng_unchanged"] and checks["finite_loss"]
    del original
    torch.cuda.empty_cache()
    write_json(output / "off_and_cache.json", checks)
    if spec["route"] == "B":
        permutation_check(model, output)
    if spec["route"] in ("A", "B"):
        for n in (0, 3, 4):
            if spec["route"] == "B":
                model.shuffle_n = n
            else:
                model.shuffle_embedding_on_test = n != 0
                model.hierarchies_to_shuffle = n or 3
            test_lifecycle(cfg, model, dm, output / f"trainer-test-{n}", batches)
            assert not model.is_testing
            write_json(output / f"trainer-test-{n}" / "intervention_trace.json", {
                "n":n, "events":getattr(model,"shuffle_events",getattr(model,"permutation_events",[])),
                "encoder_role_counts":getattr(model,"encoder_role_counts",{}),
                "shuffle_role_counts":getattr(model,"shuffle_role_counts",{}),
                "state_cleared":not model.is_testing})
            if hasattr(model,"shuffle_events"):
                model.shuffle_events=[]
        # Ordinary validation must remain OFF even after a shuffled test.
        model.eval(); model.on_validation_start()
        with torch.inference_mode():
            validation_ids, validation_scores = model.generate(**args)
        assert torch.equal(validation_ids, ids) and torch.allclose(validation_scores, scores, rtol=1e-5, atol=1e-6)
        model.on_exception(RuntimeError("bounded cleanup probe; no failed formal task"))
        assert not model.is_testing
    else:
        test_lifecycle(cfg, model, dm, output, batches)
    if spec["route"] == "A":
        from .author.checks import describe_author_model, check_original_shuffle_trace
        write_json(output / "author_structure.json", describe_author_model(model))
        write_json(output / "author_sampler_parity.json", check_original_shuffle_trace(model))
    write_json(output / "shuffle_trace.json", {"events": getattr(model, "shuffle_events", []),
                  "encoder_calls": getattr(model, "encoder_calls", getattr(model, "encoder_call_count", None)),
                  "encoder_role_counts": getattr(model, "encoder_role_counts", {}),
                  "shuffle_role_counts": getattr(model, "shuffle_role_counts", {}),
                  "shuffle_summary": getattr(model, "shuffle_summary", {}),
                  "permutation_events": getattr(model, "permutation_events", []),
                  "native_identity": model.native_identity() if hasattr(model,"native_identity") else None,
                  "test_state_after": getattr(model, "is_testing", None)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["inspect", "smoke", "trainer-smoke", "train-smoke", "resume-smoke", "formal", "attention", "summarize"])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    output = (args.output or Path(os.environ["GRID_JOB_DIR"]) / "output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invocation.json").exists():
        raise RuntimeError("Refuse to reuse a run output identity")
    start = time.monotonic()
    write_json(output / "invocation.json", {"argv": sys.argv, "interpreter": sys.executable,
        "cwd_at_entry": os.getcwd(), "spec": spec, "spec_sha256": sha256(args.spec),
        "started_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "code_identity": json.loads((REPO / "DEPLOYMENT.json").read_text()) if (REPO / "DEPLOYMENT.json").exists() else "LOCAL_UNCOMMITTED",
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES")})
    bind_source(spec["route"])
    from .common import instantiate
    cfg, model, dm = instantiate(spec, output, checkpoint=args.phase not in ("train-smoke", "resume-smoke"))
    write_json(output / "environment.json", {"versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "lightning", "hydra-core", "tensorflow-cpu", "numpy")},
        "imports": {k: str(v.__file__) for k,v in sys.modules.copy().items() if k in ("src", "src.models.modules.base_module", "src.data.loading.components.collate_functions", "src.models.modules.huggingface.transformer_base_module", "src.models.modules.semantic_id.tiger_generation_model", "src.models.modules.semantic_id.tiger_generation_model_shuffle", "src.models.modules.semantic_id.tiger_generation_model_mlp")}})
    if args.phase in ("inspect", "smoke"):
        batches = check_data(spec, cfg, output)
        from .data_checks import prepare_fixed_attention_samples, scan_test_coverage
        prepare_fixed_attention_samples(cfg, spec, output)
        coverage = scan_test_coverage(cfg, spec, output)
        truncated = coverage.get("first_truncated_engineering_example")
        if truncated is not None:
            import torch
            from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
            x = SequentialModelInputData(user_id_list=[truncated["sample_key"]],
                transformed_sequences={"sequence_data":torch.tensor([truncated["input_ids"]],dtype=torch.long)},
                mask=torch.tensor([truncated["attention_mask"]],dtype=torch.long))
            y = SequentialModuleLabelData(labels={"sequence_data":torch.tensor(truncated["target"],dtype=torch.long)})
            batches.append((x,y))
        if args.phase == "smoke":
            smoke(spec, cfg, model, dm, output, batches)
    elif args.phase == "trainer-smoke":
        cfg.trainer.limit_test_batches = 3
        test_lifecycle(cfg, model, dm, output)
    elif args.phase == "formal":
        if not spec.get("formal_approved") or not spec.get("nonauthor_gate"):
            raise RuntimeError("Formal run requires explicit approved conditions and a nonauthor gate")
        forbidden = ("fast_dev_run", "limit_test_batches", "limit_val_batches", "limit_train_batches")
        assert not any(k in cfg.trainer for k in forbidden)
        test_lifecycle(cfg, model, dm, output)
    elif args.phase in ("train-smoke", "resume-smoke"):
        from .training import train_check
        train_check(spec, cfg, model, dm, output, resume=args.phase == "resume-smoke")
    elif args.phase == "attention":
        from .observation import run_observation
        run_observation(spec, cfg, model, dm, output)
    else:
        raise NotImplementedError("This phase is not yet admitted")
    import torch
    write_json(output / "RESULT.json", {"status": "PASS", "phase": args.phase,
        "elapsed_seconds": time.monotonic() - start, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "formal": args.phase == "formal"})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
