"""Bounded real 7B verification of native eager inference."""

import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

import torch

from .attention import allowed_edges, install_attention
from .evaluate import (
    REPLAY_DIR, generation_settings, load_inference, paired_inputs, predict, read, read_replay,
)
from .identity import atomic_json, verify_adapter_identity
from .inference_attention import install_eager_inference
from .regression import absolute_errors, compare_real_contexts, original_beam
from lc_rec.shuffle_v2 import beauty
from lc_rec.shuffle_v2.replay import frozen


PREVIOUS_TRAINING = Path(
    "/data/bwwu/lc-rec/beauty-history-attention-20261127/releases/"
    "b9dfa8f8adad6ee89c15bda9d978c5465ac7e772/lc_rec/beauty_history_attention/attention.py"
)
PREVIOUS_ATTENTION_SHA256 = "9f93c4a6d6cad25c1448113011af120be145535018c793e4afa00cc3a0b4cda8"


def same_training_path():
    """Compare the default path with the verified frozen training installer."""
    from transformers import LlamaConfig, LlamaForCausalLM

    assert frozen.sha(PREVIOUS_TRAINING) == PREVIOUS_ATTENTION_SHA256
    spec = importlib.util.spec_from_file_location("previous_attention", PREVIOUS_TRAINING)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    torch.manual_seed(123)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=256,
                      pad_token_id=0)
    old = LlamaForCausalLM(cfg)
    new = LlamaForCausalLM(cfg)
    new.load_state_dict(old.state_dict())
    previous.install_attention(old, "bidirectional")
    install_attention(new, "bidirectional")
    ids = torch.tensor([[1, 3, 4, 11, 5, 6, 12, 7, 8, 2, 0],
                        [1, 4, 3, 12, 6, 5, 11, 8, 7, 2, 0]])
    valid = ids.ne(0)
    h = torch.tensor([[0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool).expand_as(ids)
    target = ids.masked_fill((torch.arange(ids.shape[1]) < 7) | ~valid, -100)

    def run(model):
        model.gradient_checkpointing_enable()
        model.train()
        results = model(ids, attention_mask=valid, history_sid_mask=h, labels=target, use_cache=False)
        results.loss.backward()
        return results.logits.detach(), results.loss.detach(), tuple(p.grad for p in model.parameters())

    x, xl, xg = run(old)
    y, yl, yg = run(new)
    assert torch.equal(x, y) and torch.equal(xl, yl)
    assert all(torch.equal(a, b) for a, b in zip(xg, yg, strict=True))
    return {"previous_source_sha256": PREVIOUS_ATTENTION_SHA256, "synthetic_cpu": True,
            "gradient_checkpointing": True, "all_weights_gradients_exact": True,
            "logits_and_loss_exact": True, "gradient_tensors": len(xg)}


def sample(dataset, tokenizer, collator, rows, row_idx):
    entry = dataset[row_idx]
    batch, targets = collator([entry])
    ids, valid = (batch[k].cuda() for k in ("input_ids", "attention_mask"))
    vocab = tokenizer.get_added_vocab()
    groups = [[vocab[part] for part in re.findall(r"<[^<>]+>", item)]
              for item in dataset.inter_data[row_idx]["inters"].split(", ")]
    history, _ = paired_inputs(ids[0].tolist(), valid[0].tolist(), groups, vocab.values(), rows[row_idx])
    h = torch.tensor([history], dtype=torch.bool, device=ids.device)
    assert targets[0] == rows[row_idx]["target"]
    return ids, valid, h, targets[0]


def install(model, mode):
    base = model.get_base_model()
    report = install_eager_inference(base, mode)
    model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
    return report


def causal_probe(base, model, tokenizer, source, allowed, settings, cases):
    old = []
    with torch.no_grad():
        for ids, valid, h, _ in cases:
            padded = torch.nn.functional.pad(ids, (3, 2))
            padmask = torch.nn.functional.pad(valid, (3, 2))
            positions = padmask.cumsum(-1) - 1
            positions.masked_fill_(padmask == 0, 1)
            raw = base(padded, attention_mask=padmask, position_ids=positions, use_cache=False).logits.cpu()
            beam = original_beam(model, tokenizer, source, allowed, ids, valid, settings)
            old.append((padded, padmask, positions, raw, beam))
        attention = install(model, "causal")
        findings = []
        for (ids, valid, h, _), (padded, padmask, positions, original, beam) in zip(cases, old, strict=True):
            extra_h = torch.nn.functional.pad(h, (3, 2))
            adapted = base(padded, attention_mask=padmask, position_ids=positions,
                           history_sid_mask=extra_h, use_cache=False).logits.cpu()
            assert torch.equal(original, adapted), absolute_errors(original, adapted)
            candidates, order, sec, boundary = predict(
                model, base, tokenizer, source, allowed, ids, valid, h, settings, True)
            comparable = [{k: row[k] for k in ("generated_token_ids", "sid", "sequence_score")}
                          for row in candidates]
            assert comparable == beam
            assert order == sorted(range(len(beam)), key=lambda i: beam[i]["sequence_score"], reverse=True)
            findings.append({"logits": absolute_errors(original, adapted), "original": beam,
                             "adapted": candidates, "stable_order": order, "actual_boundary": boundary,
                             "original_input_ids": ids[0].tolist(), "seconds": sec})
    assert attention["cache_reorders"] == 5 * len(cases), attention
    assert attention["bidirectional_mask_calls"] == 0
    return {"same_weight_same_native_path_exact": True, "cases": findings, "attention": attention}


def verify_weights(base, ids, valid, h):
    """Inspect native probabilities from every layer against the permitted graph."""
    result = base(ids, attention_mask=valid, history_sid_mask=h, output_attentions=True, use_cache=False)
    q = ids.shape[1]
    allowed = allowed_edges(valid.bool(), h, q, "bidirectional")[0]
    ordinary = allowed_edges(valid.bool(), h, q, "causal")[0]
    active = valid[0, :, None].bool().expand_as(allowed)
    reverse = allowed & ~ordinary
    groups = h[0].nonzero().flatten()
    assert groups.numel() >= 8
    early, later = int(groups[0]), int(groups[-1])
    assert reverse[early, later]
    rows = []
    for layer, weights in enumerate(result.attentions):
        probabilities = weights[0]
        assert torch.isfinite(probabilities).all()
        assert torch.count_nonzero(probabilities[:, active & ~allowed]) == 0
        positives = int(torch.count_nonzero(probabilities[:, reverse]))
        pairs = int(torch.count_nonzero(probabilities[:, early, later]))
        assert positives > 0 and pairs > 0, layer
        rows.append({"layer": layer, "shape": list(weights.shape), "dtype": str(weights.dtype),
                     "positive_reverse_history_weights": positives,
                     "cross_item_heads_with_positive_weight": pairs})
    return {"allowed_edge_count": int(allowed.sum()), "added_edges": int(reverse.sum()),
            "checked_cross_item_edge": [early, later], "layers": rows}


def b_probe(base, model, tokenizer, source, allowed, settings, case):
    ids, valid, h, target = case
    attention = install(model, "bidirectional")
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    assert len(target_ids) == 5
    suffix = torch.tensor([target_ids], device=ids.device)
    full_ids = torch.cat((ids, suffix), dim=1)
    full_valid = torch.nn.functional.pad(valid, (0, suffix.shape[1]), value=1)
    full_h = torch.nn.functional.pad(h, (0, suffix.shape[1]))
    first_changed_target_index = ids.shape[1] + 2
    changed = full_ids.clone()
    changed[0, first_changed_target_index] = suffix[0, 0]
    assert not torch.equal(changed, full_ids)

    with torch.no_grad():
        padded_ids = torch.nn.functional.pad(full_ids, (4, 3))
        padded_valid = torch.nn.functional.pad(full_valid, (4, 3))
        padded_h = torch.nn.functional.pad(full_h, (4, 3))
        observed_mask = verify_weights(base, padded_ids, padded_valid, padded_h)
        original = base(full_ids, attention_mask=full_valid, history_sid_mask=full_h,
                        output_hidden_states=True, use_cache=False)
        perturbed = base(changed, attention_mask=full_valid, history_sid_mask=full_h,
                         output_hidden_states=True, use_cache=False)
        assert torch.equal(original.logits[:, :first_changed_target_index],
                           perturbed.logits[:, :first_changed_target_index])
        assert all(torch.equal(i[full_h], j[full_h])
                   for i, j in zip(original.hidden_states, perturbed.hidden_states, strict=True))
        del original, perturbed
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            prefill = base(ids, attention_mask=valid, history_sid_mask=h, use_cache=True)
            mask = full_valid[:, :ids.shape[1]+1]
            cached = base(suffix[:, :1], attention_mask=mask, history_sid_mask=h,
                          past_key_values=prefill.past_key_values, use_cache=True).logits[:, -1]
            uncached = base(full_ids[:, :ids.shape[1]+1], attention_mask=mask,
                            history_sid_mask=h, use_cache=False).logits[:, -1]
            torch.cuda.synchronize()
        operators = sorted({row.key for row in prof.key_averages()
                            if any(k in row.key.lower() for k in ("softmax", "bmm", "sdpa", "attention"))})
        assert "aten::_softmax" in operators and "aten::bmm" in operators
        assert all("scaled_dot_product" not in op and "efficient_attention" not in op for op in operators)
        cache_error = absolute_errors(uncached, cached)
        assert cache_error["rms"] / max(cache_error["ref_rms"], 1e-8) < .03
        del prefill, cached, uncached
        references = compare_real_contexts(base, ids, valid, h)
        candidates, order, sec, boundary = predict(
            model, base, tokenizer, source, allowed, ids, valid, h, settings, True)
    assert attention["cache_reorders"] == 5
    return {"actual_layer_weights": observed_mask, "operators": operators, "cache_logits": cache_error,
            "eager_vs_explicit_fp32_contexts": references, "earlier_logits_exact": True,
            "all_layer_history_states_exact": True, "future_changed_at": first_changed_target_index,
            "actual_beam_candidates": candidates, "stable_order": order, "beam_boundary": boundary,
            "beam_generation_s": sec, "attention": attention}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("contract", "old-causal", "b-mask"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if (args.kind == "b-mask") != (args.adapter is not None):
        parser.error("give the verified preflight B adapter for b-mask only")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    report = {"kind": args.kind, "argv": sys.argv, "interpreter": sys.executable,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.kind == "contract":
        report["training_regression"] = same_training_path()
    else:
        torch.cuda.reset_peak_memory_stats()
        config = read(beauty.CONFIG)
        baseline = config["baseline"]
        _, rows, _ = read_replay(REPLAY_DIR, config)
        source = beauty.frozen_dependency(config, Path(baseline["entry"]))
        author = source.author_source(baseline["source-root"])
        index, _, _, _ = source.catalog_and_users(baseline["data-root"])
        adapter = Path(baseline["adapter"]) if args.kind == "old-causal" else args.adapter
        if args.kind == "old-causal":
            assert frozen.sha(adapter / "adapter_model.bin") == baseline["adapter_identity"]["sha256"]
        else:
            report["adapter_identity"] = verify_adapter_identity(adapter, scope="preflight", mode="bidirectional")
        model, tokenizer = load_inference(adapter, report, config, source, author, index)
        base = model.get_base_model()
        sys.path.insert(0, str(author))
        from collator import TestCollator
        from data import SeqRecDataset

        spec = SimpleNamespace(dataset="Beauty", data_path=baseline["data-root"], index_file=".index.json",
                               add_prefix=False, max_his_len=20, his_sep=", ")
        dataset = SeqRecDataset(spec, mode="test", sample_num=-1)
        dataset.set_prompt(0)
        collator = TestCollator(spec, tokenizer)
        settings, report["generation"] = generation_settings(model)
        allowed = dataset.get_prefix_allowed_tokens_fn(tokenizer)
        report["row_ids"] = [0, 8, 734] if args.kind == "old-causal" else [8]
        cases = [sample(dataset, tokenizer, collator, rows, i) for i in report["row_ids"]]
        policies = (model, tokenizer, source, allowed, settings)
        report["actual_checks"] = (causal_probe(base, *policies, cases) if args.kind == "old-causal"
                                   else b_probe(base, *policies, cases[0]))
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    report.update(status="PASS", seconds=time.monotonic()-started)
    atomic_json(args.output, report)
    print(f"{args.kind}: PASS in {report['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
