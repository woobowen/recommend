"""Actual-generation attention checks and approved finite sample collection."""
import json
from pathlib import Path

from .common import write_json, batch_keys


def capture_batch(spec, model, batch, output, engineering=True, selected_indices=None):
    import hashlib
    import numpy as np
    import torch
    from .attention import (GenerationAttentionObserver, build_encoder_position_map,
                            recover_history_item_ids, save_bundle)
    from .attention.plot import plot_bundle
    from .cli import model_arguments, numeric_difference
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.is_testing = True
    if hasattr(model, "capture_full_permutation"):
        model.capture_full_permutation = True
    if hasattr(model, "permutation_trace_limit"):
        model.permutation_trace_limit = len(model.permutation_events) + 1
    keys = batch_keys(batch)
    selected_indices = list(range(len(keys))) if selected_indices is None else selected_indices
    def parameters_hash():
        h = hashlib.sha256()
        for name, p in model.named_parameters():
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before_hash = parameters_hash()
    arguments = model_arguments(batch)
    cpu_initial = torch.get_rng_state(); cuda_initial = torch.cuda.get_rng_state()
    overrides_before = {name: vars(model).get(name) for name in ("encoder_forward_pass", "_beam_search_one_step")}
    stack = model.decoder.decoder
    hooks_before = (tuple(stack._forward_hooks), tuple(stack._forward_pre_hooks))
    with torch.inference_mode():
        off_ids, off_scores = model.generate(**arguments)[:2]
    off_cpu, off_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.set_rng_state(cpu_initial); torch.cuda.set_rng_state(cuda_initial)
    with torch.inference_mode(), GenerationAttentionObserver(model, sample_keys=keys, selected_indices=selected_indices) as observer:
        ids, scores = model.generate(**arguments)[:2]
        perm = getattr(model, "last_full_permutation", None)
        if hasattr(model, "permutation_events") and getattr(model, "shuffle_embedding_on_test", False):
            perm = torch.as_tensor(model.permutation_events[-1]["permutation"])
        n = spec.get("shuffle_n", 0)
        source = None
        if n:
            assert perm is not None, "Full actual generate-call permutation identity is required"
            full = torch.arange(4).expand(perm.shape[0], perm.shape[1], 4).clone()
            full[..., :n] = perm
            source = full[selected_indices].numpy()
        sid = torch.load(spec["sid"]["path"], map_location="cpu", weights_only=True).numpy()
        history_items = recover_history_item_ids(observer.input_sids, observer.input_mask, sid)
        positions = build_encoder_position_map(observer.input_sids, observer.input_mask,
            num_hierarchies=4, sep_enabled=model.sep_token is not None,
            user_enabled=model.user_embedding is not None and arguments["user_id"] is not None,
            aggregated=spec["route"] == "MLP", item_ids=history_items,
            source_permutations=source, num_embeddings_per_hierarchy=model.num_embeddings_per_hierarchy)
        bundle = observer.finish(ids, scores, metadata={"spec": spec, "engineering_only": engineering,
            "shuffle_n": n, "source_permutations": source.tolist() if source is not None else None,
            "source": "actual generation, identical RNG starting state for OFF/ON engineering comparison",
            "parameter_sha256": before_hash}, position_map=positions)
    checks = {"sid_exact": torch.equal(ids, off_ids), "scores": numeric_difference(scores, off_scores),
        "cpu_rng_same": torch.equal(off_cpu, torch.get_rng_state()), "cuda_rng_same": torch.equal(off_cuda, torch.cuda.get_rng_state()),
        "parameter_sha256_unchanged": parameters_hash() == before_hash,
        "model_overrides_restored": all(vars(model).get(k) is v for k,v in overrides_before.items()),
        "stack_hooks_restored": hooks_before == (tuple(stack._forward_hooks), tuple(stack._forward_pre_hooks))}
    write_json(output / "off_on_checks.json", checks)
    assert checks["sid_exact"] and checks["scores"]["exact"]
    assert all(checks[k] for k in ("cpu_rng_same", "cuda_rng_same", "parameter_sha256_unchanged", "model_overrides_restored", "stack_hooks_restored"))
    # Exception cleanup is checked without invoking encoder or consuming RNG.
    try:
        with GenerationAttentionObserver(model, sample_keys=keys, selected_indices=selected_indices):
            raise RuntimeError("intentional observer cleanup probe")
    except RuntimeError as error:
        assert str(error) == "intentional observer cleanup probe"
    assert hooks_before == (tuple(stack._forward_hooks), tuple(stack._forward_pre_hooks))
    assert all(vars(model).get(k) is v for k,v in overrides_before.items())
    saved = save_bundle(bundle, output / "bundle")
    plotted = plot_bundle(output / "bundle" / "record.json", output / "figures")
    model.is_testing = False
    if hasattr(model, "capture_full_permutation"):
        model.capture_full_permutation = False
    write_json(output / "RESULT.json", {"status": "PASS", "saved": saved,
        "png_count": len(plotted["products"]), "engineering_only": engineering,
        "exception_cleanup": True, "scientific_similarity_claim": None})
    return checks


def run_observation(spec, cfg, model, dm, output):
    import torch
    from .common import batch_to, make_loader
    model.to("cuda:0").eval()
    batch = batch_to(next(iter(make_loader(cfg, "val", limit_rows=3))), "cuda:0")
    capture_batch(spec, model, batch, output / "small-attention", engineering=True)
