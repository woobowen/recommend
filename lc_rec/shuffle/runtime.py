"""Token intervention and bounded GPU evidence for the frozen inference entries."""
import hashlib
import json
from pathlib import Path
import sys
from core import transform

OPTIONS = None
CPU = None
CONTEXT = None
TOKENIZER = None


def configure(options):
    global OPTIONS, CPU
    OPTIONS = options
    CPU = json.loads(Path(options.cpu_audit).read_text())
    assert CPU['status'] == 'PASS_CPU_FULL_INPUTS' and CPU['model_key'] == options.model_key


def preflight_assignment(rank):
    return CPU['preflight_assignments'][str(rank)]


def intervene(inputs, index, data, tokenizer):
    global CONTEXT
    import torch
    ids = inputs['input_ids'][0].tolist()
    mask = inputs['attention_mask'][0].tolist()
    uid = list(data.inters)[index]
    seq = data.inters[uid]
    groups = [[CPU['token_ids'][t] for t in data.indices[str(item)]] for item in seq[:-1][-20:]]
    original = tokenizer(data[index]['input_ids'], return_attention_mask=True)
    assert ids == original['input_ids'] and mask == original['attention_mask']
    target = data[index]['labels']
    assert target == ''.join(data.indices[str(seq[-1])])
    new, evidence = transform(ids, mask, groups, set(CPU['token_ids'].values()), OPTIONS.shuffle_n, index)
    record = {'n': OPTIONS.shuffle_n, 'seed': 42, 'protocol': 'GRID-LETTER-SHUFFLE-V1',
              'row_idx': index, 'original_input_token_ids': ids, 'attention_mask': mask,
              'evidence': evidence, 'cpu_manifest_sha256': CPU['manifest']['sha256']}
    inputs['input_ids'] = torch.tensor([new], dtype=inputs['input_ids'].dtype,
                                     device=inputs['input_ids'].device)
    CONTEXT = dict(record=record, groups=groups, target=target, text=data[index]['input_ids'])
    return inputs, record


class ShuffleAllowed:
    """Retain the original Allowed checks; bind the validated current prefix."""
    def __init__(self, tokenizer, indices):
        from common import Allowed
        self.original = Allowed(tokenizer, indices)

    def bind(self, prompt_ids, text, record):
        # Original bind verifies the full untampered text, offsets and delimiter.
        raw = record['original_input_token_ids']
        self.original.bind(raw, text)
        assert record['row_idx'] == CONTEXT['record']['row_idx']
        expected, _ = transform(raw, record['attention_mask'], CONTEXT['groups'],
                                set(CPU['token_ids'].values()), record['n'], record['row_idx'])
        assert list(prompt_ids) == expected
        sep = self.original.sep
        assert list(prompt_ids)[-len(sep):] == sep
        self.original.prefix = list(prompt_ids)

    def __call__(self, batch_id, sentence):
        return self.original(batch_id, sentence)


def loaded(model, tokenizer, args, report):
    global TOKENIZER
    TOKENIZER = tokenizer
    report['shuffle'] = {'n': OPTIONS.shuffle_n, 'seed': 42, 'protocol': 'GRID-LETTER-SHUFFLE-V1',
                         'cpu_manifest_sha256': CPU['manifest']['sha256'],
                         'intervention': 'complete original tokenization then within-history-item Token ID permutation',
                         'no_explicit_old_positions_or_prefilled_cache': True}
    if OPTIONS.acceptance:
        assert OPTIONS.shuffle_n == 0 and args.preflight_users_per_rank == 16
        install_acceptance(model, tokenizer, args, report)


def install_acceptance(model, tok, args, report):
    import torch
    import time
    import os
    rank = int(os.environ['RANK'])
    out = Path(args.output)
    generate = model.generate
    historical = {}
    root = CPU['normal_test_dir']
    wanted = set(preflight_assignment(rank))
    # Read only once. Historical rows are not rewritten or regenerated in full.
    for shard in [0, 1]:
        with (Path(root) / ('predictions.rank%d.jsonl' % shard)).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['row_idx'] in wanted:
                    historical[row['row_idx']] = row
    assert set(historical) == wanted
    evidence = []
    # Declared before inference. Same-process OFF must be exact; historical BF16
    # scores allow 1e-5 absolute/relative, with exact sequences and ordering.
    report['historical_comparison_tolerance'] = {'atol': 1e-5, 'rtol': 1e-5,
                                               'sequences_and_order': 'exact'}

    def wrapped(**kwargs):
        ctx = CONTEXT
        row_idx = ctx['record']['row_idx']
        raw = ctx['record']['original_input_token_ids']
        mask = ctx['record']['attention_mask']
        assert set(kwargs) <= {'input_ids', 'attention_mask', 'max_new_tokens',
             'prefix_allowed_tokens_fn', 'do_sample', 'use_cache', 'num_beams',
             'num_return_sequences', 'output_scores', 'return_dict_in_generate',
             'early_stopping', 'eos_token_id', 'pad_token_id'}
        assert kwargs['input_ids'].tolist() == [raw]
        assert kwargs['attention_mask'].tolist() == [mask]
        assert kwargs['num_beams'] == kwargs['num_return_sequences'] == 20
        receipt = {'row_idx': row_idx, 'conditions': {}, 'cache_position': []}
        generated_by_condition = {}
        for label, n in [('original', 0), ('OFF', 0), ('S3', 3), ('S4', 4)]:
            ids, ev = transform(raw, mask, ctx['groups'], set(CPU['token_ids'].values()), n, row_idx)
            new = torch.tensor([ids], dtype=kwargs['input_ids'].dtype, device=kwargs['input_ids'].device)
            callback = kwargs['prefix_allowed_tokens_fn']
            if isinstance(callback, ShuffleAllowed):
                record = dict(ctx['record'], n=n)
                callback.bind(ids, ctx['text'], record)
            # Verify the actual active lookup layer, before RoPE/attention.
            with torch.no_grad():
                emb = model.get_input_embeddings()
                original_emb = emb(kwargs['input_ids'])
                lookup = emb(new)
                sid_positions = [i for i, token in enumerate(raw) if token in set(CPU['token_ids'].values())]
                old_ids = [raw[i] for i in sid_positions]
                for j, start in enumerate(sid_positions[::4]):
                    # Use the ID identity within this four-token group; layers
                    # are distinct registered symbols even when numbers match.
                    old = raw[start:start+4]
                    mapping = [old.index(token) for token in ids[start:start+4]]
                    assert torch.equal(lookup[:, start:start+4], original_emb[:, start:start+4][:, mapping])
                outside = [i for i in range(len(raw)) if i not in sid_positions]
                assert torch.equal(lookup[:, outside], original_emb[:, outside])
            calls = []
            def before_forward(module, positional, named):
                if calls:
                    return
                actual_ids = named.get('input_ids', positional[0] if positional else None)
                assert actual_ids is not None and actual_ids[0].tolist() == ids
                past = named.get('past_key_values')
                if past is None:
                    old_length = 0
                elif hasattr(past, 'get_seq_length'):
                    old_length = int(past.get_seq_length())
                else:
                    old_length = int(past[0][0].shape[-2]) if len(past) else 0
                assert old_length == 0, 'First prefill must have an empty cache'
                pos = named.get('position_ids')
                expected_pos = kwargs['attention_mask'].long().cumsum(-1)-1
                expected_pos.masked_fill_(kwargs['attention_mask'] == 0, 1)
                assert pos is not None and torch.equal(pos[0], expected_pos[0])
                calls.append({'initial_cache_length': old_length, 'position_ids': pos[0].tolist(),
                              'current_input_ids': actual_ids[0].tolist()})
            handle = model.get_base_model().model.register_forward_pre_hook(before_forward, with_kwargs=True)
            try:
                tick = time.monotonic()
                generated = generate(**dict(kwargs, input_ids=new))
                torch.cuda.synchronize()
            finally:
                handle.remove()
            assert calls
            sequences = generated.sequences.cpu().tolist()
            scores = generated.sequences_scores.float().cpu()
            suffixes = [seq[len(ids):] for seq in sequences]
            assert all(tok.eos_token_id in s and s.index(tok.eos_token_id) == 4 for s in suffixes)
            for suffix in suffixes:
                for position, token in enumerate(suffix[:5]):
                    assert token in callback(0, torch.tensor(ids+suffix[:position], dtype=torch.long))
            generated_by_condition[label] = (suffixes, scores)
            receipt['conditions'][label] = {'input_ids': ids, 'evidence': ev,
                'sequences': suffixes, 'scores': scores.tolist(), 'lookup_equivalence_exact': True,
                'prefill': calls[0], 'generation_seconds': time.monotonic()-tick}
            if label == 'OFF':
                returned = generated
            else:
                del generated
        assert generated_by_condition['original'][0] == generated_by_condition['OFF'][0]
        assert torch.equal(generated_by_condition['original'][1], generated_by_condition['OFF'][1])
        old = historical[row_idx]
        assert old['input_token_ids'] == raw
        expected_suffixes = [c['generated_token_ids'] for c in old['candidates']]
        expected_scores = torch.tensor([c['sequence_score'] for c in old['candidates']])
        assert generated_by_condition['OFF'][0] == expected_suffixes
        assert torch.allclose(generated_by_condition['OFF'][1], expected_scores, atol=1e-5, rtol=1e-5)
        actual_scores = generated_by_condition['OFF'][1].tolist()
        assert sorted(range(20), key=lambda i: actual_scores[i], reverse=True) == old['sorted_generation_order']
        receipt['same_process_OFF_exact'] = True
        receipt['historical_sequences_scores_order_PASS'] = True
        evidence.append(receipt)
        (out / ('acceptance.rank%d.json' % rank)).write_text(json.dumps(
            {'status': 'PASS' if len(evidence) == 16 else 'RUNNING', 'users': len(evidence),
             'samples': evidence}, indent=2) + '\n')
        if isinstance(kwargs['prefix_allowed_tokens_fn'], ShuffleAllowed):
            kwargs['prefix_allowed_tokens_fn'].bind(raw, ctx['text'], ctx['record'])
        return returned
    model.generate = wrapped
