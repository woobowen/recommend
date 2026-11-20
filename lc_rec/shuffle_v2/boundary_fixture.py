"""Exercise the V2 hooks on a tiny synthetic CPU LLaMA, using the frozen APIs."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.replay import dump, semantic_hash
from lc_rec.shuffle_v2.runtime import Runtime


def main():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(23)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=4, eos_token_id=2, pad_token_id=0))
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=2,
                                           inference_mode=True, target_modules=['q_proj'])).eval()
    active_group = [40, 45, 50, 55]
    tokens = {f'token_{i}': i for i in active_group}
    tokenizer = SimpleNamespace(get_vocab=lambda: tokens, encode=lambda s, add_special_tokens: [tokens[s]])
    original = [0, 1, 10, *active_group, 12]
    mask = [0] + [1] * (len(original) - 1)
    runtime = Runtime.__new__(Runtime)
    runtime.options = SimpleNamespace(model_key='synthetic_fixture', shuffle_mode='shuffle4')
    runtime.audit = {'token_ids': tokens, 'tokenizer_vocab_sha256': semantic_hash(tokens)}
    runtime.digest = 'synthetic-fixture'
    report = {}
    runtime.loaded(model, tokenizer, SimpleNamespace(preflight_users_per_rank=1), report)
    results = {}
    for mode in core.MODES:
        new_ids, evidence = core.transform(original, mask, [active_group], active_group, mode, 5)
        record = {}
        runtime.context = new_ids, mask, torch.tensor([original]), evidence, record

        def callback(_, sequence):
            offset = len(sequence) - len(original)
            return [2] if offset == 4 else list(range(40 + 5 * offset, 45 + 5 * offset))

        with torch.no_grad():
            result = model.generate(input_ids=torch.tensor([new_ids]), attention_mask=torch.tensor([mask]),
                                    max_new_tokens=10, prefix_allowed_tokens_fn=callback, num_beams=20,
                                    num_return_sequences=20, output_scores=True, return_dict_in_generate=True,
                                    early_stopping=True)
        assert result.sequences.shape == (20, len(original) + 5)
        assert result.sequences[:, -1].tolist() == [2] * 20
        observed = record['actual_prefill']
        assert observed['lookup_exact'] and observed['input_ids'] == new_ids and observed['attention_mask'] == mask
        assert observed['position_ids'] == [1] + list(range(len(original) - 1))
        results[mode] = observed
    dump(args.output / 'summary.json', {'status': 'PASS_SYNTHETIC_CPU_BOUNDARY', 'results': results,
                                       'stage': 'synthetic 1-layer CPU model, no A0 checkpoint or real-user beams'})


if __name__ == '__main__':
    main()
