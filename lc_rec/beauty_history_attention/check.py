"""Bounded checks of real LETTER inputs and the layered mask implementation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from lc_rec.beauty_history_attention.attention import allowed_edges, install_attention
from lc_rec.beauty_history_attention.inputs import HistoryCollator


def real_inputs(args):
    sys.path.insert(0, str(args.source_root))
    from collator import Collator, TestCollator
    from data import SeqRecTestDataset
    from utils import load_datasets, set_seed

    set_seed(42)
    cfg = SimpleNamespace(dataset="Beauty", data_path=str(args.data_root), index_file=".index.json",
                          tasks="seqrec", max_his_len=20, his_sep=", ", add_prefix=False,
                          sample_valid=True, only_train_response=True, valid_prompt_id=0,
                          valid_prompt_sample_num=2, train_prompt_sample_num="1",
                          train_data_sample_num="0")
    train, valid = load_datasets(cfg)
    assert len(train) == 131413 and len(valid) == 44726
    train = train.datasets[0]
    test = SeqRecTestDataset(cfg, prompt_id=0)
    tokenizer = LlamaTokenizer.from_pretrained(args.model, model_max_length=2048,
                                               padding_side="right", local_files_only=True)
    tokenizer.pad_token_id = 0
    sid = train.get_new_tokens()
    assert tokenizer.add_tokens(sid) == 965 and len(tokenizer) == 32965
    collator = Collator(cfg, tokenizer)
    history_collator = HistoryCollator(collator, tokenizer, sid)
    long_row = next(j for j, d in enumerate(train.inter_data) if len(d["inters"].split(", ")) == 20)
    examples = [dict(zip(("input_ids", "labels"), train._get_text_data(train.inter_data[long_row], template)))
                for template in train.prompts]
    examples += [valid[i] for i in range(32)]
    probe = {}
    tokens = set(history_collator.sid.values())
    assert len(train.prompts) == 17
    for side in ("right", "left"):
        tokenizer.padding_side = side
        legacy = collator(examples)
        actual = history_collator(examples)
        assert all(torch.equal(legacy[field], actual[field]) for field in legacy)
        mark = actual["history_sid_mask"]
        assert mark[:17].sum(1).tolist() == [80] * 17
        assert not (mark & ~actual["attention_mask"].bool()).any()
        for i in range(len(examples)):
            sid_locations = [p for p, code in enumerate(actual["input_ids"][i].tolist()) if code in tokens]
            assert all(mark[i, p] for p in sid_locations[:-4])
            assert not mark[i, sid_locations[-4:]].any()
        if side == "right":
            assert ((actual["labels"] != -100).sum(1) == 5).all()
        probe[side] = {"nonpad_lengths": actual["attention_mask"].sum(1).tolist(),
                       "history_lengths": mark.sum(1).tolist(),
                       "supervised_lengths": (actual["labels"] != -100).sum(1).tolist()}
    tokenizer.padding_side = "right"
    full = history_collator(examples[:1])
    first = full["history_sid_mask"][0].nonzero().squeeze(1)[0].item()
    truncated = []
    for side, limit in [("right", first - 1), ("right", first + 2), ("left", 31)]:
        tokenizer.truncation_side = side
        tokenizer.model_max_length = limit
        actual = history_collator(examples[:1])
        legacy = collator(examples[:1])
        assert all(torch.equal(legacy[field], actual[field]) for field in legacy)
        # The tokenizer truncates ordinary tokens and then adds BOS.
        indices = list(range(limit)) if side == "right" else [0, *range(full["input_ids"].shape[1] - limit + 1, full["input_ids"].shape[1])]
        assert torch.equal(actual["input_ids"], full["input_ids"][:, indices])
        expected = full["history_sid_mask"][:, indices]
        assert torch.equal(actual["history_sid_mask"], expected)
        truncated.append({"truncation_side": side, "cap": limit, "retained_history": int(expected.sum())})
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 2048
    test_rows = [test[i] for i in (0, 8, 47, 734)]
    test_legacy, answers = TestCollator(cfg, tokenizer)(test_rows)
    assert answers == [row["labels"] for row in test_rows]
    for i, row in enumerate(test_rows):
        expected = history_collator.history_ids(row["input_ids"])
        assert [p.item() for p in test_legacy["input_ids"][i] if p.item() in tokens] == expected
    return {"vocab": len(tokenizer), "prompts": len(train.prompts), "training_rows": len(train),
            "validation_rows": len(valid), "tested_validation_rows": 32,
            "test_rows": len(test), "tested_test_rows": [0, 8, 47, 734],
            "real_max_history_sample": long_row, "padding": probe, "artificial_caps": truncated,
            "validated_test_padding_side": tokenizer.padding_side,
            "sample": {k: v[0].tolist() for k, v in full.items()}}


def model_checks(device):
    torch.manual_seed(123)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, max_position_embeddings=256, pad_token_id=0,
                      bos_token_id=1, eos_token_id=2)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    original = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    b = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    causal = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    for model in (b, causal):
        model.load_state_dict(original.state_dict())
    b_report = install_attention(b, "bidirectional")
    causal_report = install_attention(causal, "causal")
    ids = torch.tensor([[0, 1, 3, 4, 5, 6, 7, 8, 9], [1, 10, 11, 12, 13, 14, 15, 16, 17],
                        [0] * 9], device=device)
    valid = ids.ne(0)
    h = torch.tensor([[0, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1, 0, 0, 0],
                      [0] * 9], dtype=torch.bool, device=device)
    for mode in ("bidirectional", "causal"):
        mask = allowed_edges(valid, h, ids.shape[1], mode).tolist()
        for n, row in enumerate(mask):
            for i, keys in enumerate(row):
                for j, observed in enumerate(keys):
                    assert observed == bool(valid[n, i] and valid[n, j] and (
                        j <= i or (mode == "bidirectional" and h[n, i] and h[n, j])))
    expected_atol = 0.016 if device == "cuda" else 5e-7
    with torch.no_grad():
        old = original(ids, attention_mask=valid, use_cache=False).logits
        ca = causal(ids, attention_mask=valid, history_sid_mask=h, use_cache=False).logits
        torch.testing.assert_close(old[valid], ca[valid], atol=expected_atol, rtol=0)
        start = b(ids, attention_mask=valid, history_sid_mask=h, output_hidden_states=True, use_cache=False)
        changed = ids.clone()
        changed[:2, 8] += 1
        end = b(changed, attention_mask=valid, history_sid_mask=h, output_hidden_states=True, use_cache=False)
        assert all(torch.equal(x[:, :8], y[:, :8]) for x, y in zip(start.hidden_states, end.hidden_states))
        assert torch.equal(start.logits[:, :8], end.logits[:, :8])
        assert torch.isfinite(start.logits).all()
        change_history = ids.clone()
        change_history[0, 6] += 8
        late = b(change_history, attention_mask=valid, history_sid_mask=h, use_cache=False).logits
        assert not torch.equal(late[0, 2], start.logits[0, 2])
        full = b(ids[:2], attention_mask=valid[:2], history_sid_mask=h[:2], use_cache=False).logits
        pref = b(ids[:2, :8], attention_mask=valid[:2, :8], history_sid_mask=h[:2, :8], use_cache=True)
        reorder = torch.tensor([1, 0, 1], device=device)
        cache = b._reorder_cache(pref.past_key_values, reorder)
        tail = b(ids[reorder, 8:], attention_mask=valid[reorder], history_sid_mask=h[reorder, :8],
                 past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(tail[:, 0], full[reorder, -1], atol=expected_atol, rtol=0)
        try:
            causal(ids[:2, 8:], attention_mask=valid[:2], history_sid_mask=h[:2],
                   past_key_values=pref.past_key_values, use_cache=True)
        except ValueError as exc:
            assert "KV cache" in str(exc)
        else:
            raise AssertionError("cache from bidirectional model was accepted by causal mode")
        # Left-padding generation, distinct H across the two examples, 20 real beams.
        a = b.generate(ids[:2, :8], attention_mask=valid[:2, :8], history_sid_mask=h[:2, :8],
                       num_beams=20, num_return_sequences=20, max_new_tokens=2, use_cache=True)
        assert a.shape == (40, 10)
    b.train()
    label = ids.clone()
    label[:, :7] = -100
    label[~valid] = -100
    # Interleave two forward graphs with different masks before either backward.
    def gradients(checkpoint):
        b.gradient_checkpointing_enable() if checkpoint else b.gradient_checkpointing_disable()
        b.zero_grad(set_to_none=True)
        x = b(ids, attention_mask=valid, history_sid_mask=h, labels=label, use_cache=False).loss
        alternate = h.roll(1, dims=1) & valid
        y = b(ids, attention_mask=valid, history_sid_mask=alternate, labels=label, use_cache=False).loss
        (x + y).backward()
        assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
        return {n: p.grad.clone() for n, p in b.named_parameters() if p.grad is not None}

    normal_grad = gradients(False)
    checkpoint_grad = gradients(True)
    grad_diff = max((normal_grad[n] - checkpoint_grad[n]).abs().max().item() for n in normal_grad)
    for name in normal_grad:
        torch.testing.assert_close(normal_grad[name], checkpoint_grad[name], atol=expected_atol, rtol=0)
    return {"synthetic": True, "checked_b_layers": b_report, "checked_causal_layers": causal_report,
            "causal_legacy_max_abs": (old[valid] - ca[valid]).abs().max().item(),
            "causal_legacy_absolute_tolerance": expected_atol,
            "cached_max_abs": (tail[:, 0] - full[reorder, -1]).abs().max().item(),
            "checkpoint_two_outstanding_graphs_gradient_max_abs": grad_diff, "beam_output_shape": list(a.shape),
            "history_hidden_and_earlier_predictions_equal_after_final_target_change": True,
            "changed_cross_item_future_history_affects_earlier_history_query": True,
            "native_foreign_cache_rejected": True, "all_masked_row_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "data-root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {"real_inputs": real_inputs(args), "model_checks": model_checks(args.device)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
