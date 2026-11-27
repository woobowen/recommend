"""Compare eager causal inference, experimental causal, and history attention on old A weights."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import sys
import re

import torch

from lc_rec.beauty_history_attention.attention import install_attention
from lc_rec.beauty_history_attention.evaluate import (
    REPLAY_DIR, load_inference, paired_inputs, read_replay, predict, generation_settings,
)
from lc_rec.shuffle_v2 import beauty
from lc_rec.shuffle_v2.replay import frozen


def absolute_errors(reference, actual):
    reference, actual = reference.float(), actual.float()
    delta = (reference - actual).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "rms": delta.square().mean().sqrt().item(),
            "ref_rms": reference.square().mean().sqrt().item(),
            "reference_max_abs": reference.abs().max().item(),
            "same_argmax": bool(torch.equal(reference.argmax(-1), actual.argmax(-1)))}


def original_beam(model, tokenizer, source, allowed, ids, mask, settings):
    generated = model.generate(input_ids=ids, attention_mask=mask,
                               prefix_allowed_tokens_fn=allowed, **settings)
    decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
    return [{"generated_token_ids": seq[ids.shape[1]:], "sid": source.normalized_prediction(text),
             "sequence_score": score}
            for seq, score, text in zip(generated.sequences.tolist(), generated.sequences_scores.tolist(),
                                        decoded, strict=True)]


def causal_gradient(base, inputs, labels, h=None):
    # The native inference LM is used in both cases (no dropout or 8-bit
    # loading). Compare the historical LETTER loss formula on the same logits.
    names = ("layers.0.self_attn.q_proj.lora_B.default.weight",
             "layers.31.self_attn.q_proj.lora_B.default.weight")
    selected = {n: p for n, p in base.model.named_parameters() if n in names}
    assert set(selected) == set(names)
    for p in selected.values():
        p.requires_grad_(True)
    base.model.training = True
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    try:
        kwargs = {} if h is None else {"history_sid_mask": h}
        logits = base(**inputs, use_cache=False, **kwargs).logits
        target = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits[:, :-1][target != -100] / .8,
                                                   target[target != -100])
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in selected.items()}
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients.values())
        return loss.item(), gradients
    finally:
        base.disable_input_require_grads()
        base.gradient_checkpointing_disable()
        base.model.training = False
        for p in selected.values():
            p.grad = None
            p.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    stamp = time.monotonic()
    cfg = json.loads(Path("lc_rec/shuffle_v2/a0_beauty_letter.json").read_text())
    _, rows, _ = read_replay(REPLAY_DIR, cfg)
    baseline = cfg["baseline"]
    source = beauty.frozen_dependency(cfg, Path(baseline["entry"]))
    author = source.author_source(baseline["source-root"])
    index, _, _, _ = source.catalog_and_users(baseline["data-root"])
    old_adapter = Path(baseline["adapter"])
    assert frozen.sha(old_adapter / "adapter_model.bin") == baseline["adapter_identity"]["sha256"]
    report = {"source": "diagnostic on old A weights, with no optimizer or parameter update"}
    model, tokenizer = load_inference(old_adapter, report, cfg, source, author, index)
    base = model.get_base_model()
    sys.path.insert(0, str(author))
    from collator import TestCollator
    from data import SeqRecDataset
    from types import SimpleNamespace
    spec = SimpleNamespace(dataset="Beauty", data_path=baseline["data-root"], index_file=".index.json",
                           add_prefix=False, max_his_len=20, his_sep=", ")
    dataset = SeqRecDataset(spec, mode="test", sample_num=-1)
    dataset.set_prompt(0)
    collator = TestCollator(spec, tokenizer)
    row_idx = 8
    batch, targets = collator([dataset[row_idx]])
    ids, valid = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
    sid_vocab = tokenizer.get_added_vocab()
    groups = [[sid_vocab[part] for part in re.findall(r"<[^<>]+>", item)]
              for item in dataset.inter_data[row_idx]["inters"].split(", ")]
    history, _ = paired_inputs(ids[0].tolist(), valid[0].tolist(), groups, sid_vocab.values(), rows[row_idx])
    h = torch.tensor([history], dtype=torch.bool, device=ids.device)
    allowed = dataset.get_prefix_allowed_tokens_fn(tokenizer)
    settings, effective = generation_settings(model)
    report["sample"] = {"row_idx": row_idx, "input_ids": ids[0].tolist(), "H": h[0].tolist(),
                        "target": targets[0], "generation": effective}
    target = tokenizer(targets[0], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    train_ids = torch.cat((ids, torch.tensor([target], device=ids.device)), dim=1)
    train_valid = train_ids.ne(0)
    training_h = torch.nn.functional.pad(h, (0, len(target)))
    labels = train_ids.masked_fill(torch.arange(train_ids.shape[1], device=ids.device) < ids.shape[1], -100)
    original = (base.forward, base.prepare_inputs_for_generation, base._reorder_cache,
                base.model._prepare_decoder_attention_mask,
                [layer.self_attn.forward for layer in base.model.layers])
    with torch.no_grad():
        old_logits = base(train_ids, attention_mask=train_valid, use_cache=False).logits.cpu()
        old_beams = original_beam(model, tokenizer, source, allowed, ids, valid, settings)
    old_loss, old_grads = causal_gradient(base, {"input_ids": train_ids, "attention_mask": train_valid}, labels)

    ca_report = install_attention(base, "causal")
    model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
    with torch.no_grad():
        ca_logits = base(train_ids, attention_mask=train_valid, history_sid_mask=training_h, use_cache=False).logits.cpu()
        ca_beams, order, _, _ = predict(model, base, tokenizer, source, allowed, ids, valid, h, settings, True)
    ca_loss, ca_grads = causal_gradient(base, {"input_ids": train_ids, "attention_mask": train_valid},
                                       labels, training_h)
    report["eager_vs_experimental_causal"] = {
        "all_logits": absolute_errors(old_logits, ca_logits),
        "supervised_logits": absolute_errors(old_logits[:, ids.shape[1] - 1:-1],
                                             ca_logits[:, ids.shape[1] - 1:-1]),
        "temperature_0p8_shifted_ce_eager": old_loss, "temperature_0p8_shifted_ce_experimental": ca_loss,
        "gradient": {n: absolute_errors(old_grads[n], ca_grads[n]) for n in old_grads},
        "eager_candidates": old_beams, "experimental_candidates": ca_beams,
        "experimental_candidate_order": order,
        "same_candidate_order": [c["generated_token_ids"] for c in old_beams] ==
                                [c["generated_token_ids"] for c in ca_beams],
        "shared_candidate_ids": len({tuple(c["generated_token_ids"]) for c in old_beams}
                                    & {tuple(c["generated_token_ids"]) for c in ca_beams}),
        "attention": ca_report,
    }

    base.forward, base.prepare_inputs_for_generation, base._reorder_cache, base.model._prepare_decoder_attention_mask, layers = original
    for layer, method in zip(base.model.layers, layers, strict=True):
        layer.self_attn.forward = method
    del base._beauty_history_attention
    b_report = install_attention(base, "bidirectional")
    model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
    args.output.joinpath("partial-causal.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with torch.no_grad():
        profile = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA])
                   if args.profile else nullcontext())
        with profile:
            pre = base(ids, attention_mask=valid, history_sid_mask=h, use_cache=True)
            step_ids = train_ids[:, ids.shape[1]:ids.shape[1] + 1]
            next_mask = train_valid[:, :ids.shape[1] + 1]
            step = base(step_ids, attention_mask=next_mask, history_sid_mask=h,
                        past_key_values=pre.past_key_values, use_cache=True)
            no_cache = base(train_ids[:, :ids.shape[1] + 1], attention_mask=next_mask,
                            history_sid_mask=h, use_cache=False)
            torch.cuda.synchronize()
        if args.profile:
            report["observed_attention_operators"] = sorted({event.name for event in profile.events()
                                                            if "attention" in event.name.lower()
                                                            or "fmha" in event.name.lower()})
        del pre, no_cache, step
        first = base(train_ids, attention_mask=train_valid, history_sid_mask=training_h,
                     output_hidden_states=True, use_cache=False)
        changed = train_ids.clone()
        changed[0, -3] = target[0]
        assert not torch.equal(changed, train_ids)
        later = base(changed, attention_mask=train_valid, history_sid_mask=training_h,
                     output_hidden_states=True, use_cache=False)
        history_equal = all(torch.equal(a[training_h], b[training_h])
                            for a, b in zip(first.hidden_states, later.hidden_states, strict=True))
        earlier_logits_equal = torch.equal(first.logits[:, :-3], later.logits[:, :-3])
        assert history_equal and earlier_logits_equal
        del first, later
        # Retain only three vocab rows and head-independent error summaries.
        true_prefill = base(ids, attention_mask=valid, history_sid_mask=h, use_cache=True)
        cached = base(step_ids, attention_mask=next_mask, history_sid_mask=h,
                      past_key_values=true_prefill.past_key_values, use_cache=True).logits[:, -1]
        uncached = base(train_ids[:, :ids.shape[1] + 1], attention_mask=next_mask,
                        history_sid_mask=h, use_cache=False).logits[:, -1]
        report["bidirectional"] = {"earlier_logits_equal": earlier_logits_equal,
                                   "history_equal_each_layer": history_equal,
                                   "cached_vs_uncached_next_logits": absolute_errors(uncached, cached),
                                   "attention": b_report}
    report["seconds"] = time.monotonic() - stamp
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    args.output.joinpath("regression.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
