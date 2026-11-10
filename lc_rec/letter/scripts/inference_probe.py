"""Strict adapter reload and bounded original LETTER prompt0/beam20 timing.

The author's LoRA test branch reloads a BF16 base, not an int8 base. This
script preserves that branch and does not report recommendation metrics.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
import resource
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def saved_key(name):
    if '.modules_to_save.default.' in name:
        return name.replace('.modules_to_save.default.', '.')
    if '.lora_A.default.' in name or '.lora_B.default.' in name:
        return name.replace('.default.', '.')
    raise AssertionError('Unexpected trained tensor family: ' + name)


def select_users(lengths, world, np):
    """Fixed first matching real users at exact prompt0 length quantiles."""
    used, selected = set(), {}
    for kind, quantile in [('q10', .1), ('q50', .5), ('q90', .9), ('max', 1.)]:
        k = int(round(quantile * (len(lengths) - 1)))
        target = int(np.partition(lengths, k)[k])
        candidates = [int(i) for i in np.flatnonzero(lengths == target) if int(i) not in used]
        if len(candidates) < world:
            raise AssertionError('Insufficient distinct users for fixed length stratum ' + kind)
        rows = candidates[:world]
        used.update(rows)
        selected[kind] = [{'row_idx': i, 'input_length': int(lengths[i]), 'prompt_id': 0} for i in rows]
    return selected


def main():
    parser = argparse.ArgumentParser()
    for name in ['source', 'data-root', 'model', 'adapter', 'audit-output', 'train-dir', 'output', 'run-commit']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--minimum-host-free-gib', type=float, default=40.)
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema': 'letter_strict_reload_beam20_v2', 'rank': rank, 'world_size': world,
              'argv': sys.argv, 'cwd': str(Path.cwd()), 'host': socket.gethostname(),
              'interpreter': sys.executable, 'RUN_COMMIT': args.run_commit,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'status': 'RUNNING', 'no_metrics': True, 'timings': []}
    torch, dist = None, None

    def guard():
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        report['minimum_observed_host_memavailable_bytes'] = min(available, report.get('minimum_observed_host_memavailable_bytes', available))
        if available < args.minimum_host_free_gib * 1024**3:
            raise MemoryError('Host memory reserve fell below configured 40 GiB minimum')

    try:
        if world != 2:
            raise ValueError('Exactly two ranks required for dual-device inference timing')
        if args.minimum_host_free_gib < 40:
            raise ValueError('Host memory reserve cannot be lowered below 40 GiB')
        guard()
        import numpy as np
        import torch
        import torch.distributed as dist
        from transformers import LlamaForCausalLM, LlamaTokenizer
        from peft import PeftModel, get_peft_model_state_dict
        if importlib.metadata.version('peft') != '0.5.0':
            raise RuntimeError('This strict reload probe is pinned to PEFT 0.5.0')
        report['versions'] = {name: importlib.metadata.version(name) for name in
                              ['torch', 'transformers', 'peft', 'bitsandbytes', 'deepspeed', 'safetensors']}
        torch.cuda.set_device(local_rank)
        torch.manual_seed(42)
        dist.init_process_group('nccl', timeout=dt.timedelta(minutes=5))
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, args.source)
        from data import SeqRecDataset
        from collator import TestCollator
        source = Path(args.source)
        report['author_sources'] = {name: {'path': str(source / name), 'sha256': sha(source / name)}
                                    for name in ['data.py', 'collator.py', 'test_ddp.py']}
        audit = Path(args.audit_output)
        samples = json.loads((audit / 'probe_samples.json').read_text())
        for kind, identity in samples['a_data_identity'].items():
            path = Path(args.data_root) / 'Instruments' / ('Instruments.' + kind + '.json')
            if sha(path) != identity['sha256']:
                raise AssertionError('Fixed data/SID identity mismatch: ' + str(path))
        adapter = Path(args.adapter)
        required_files = ['adapter_config.json', 'tokenizer_config.json', 'tokenizer.model', 'added_tokens.json']
        if any(not (adapter / name).is_file() for name in required_files):
            raise AssertionError('Adapter tokenizer/config artifact is missing')
        bins = [adapter / name for name in ['adapter_model.bin', 'adapter_model.safetensors'] if (adapter / name).is_file()]
        if len(bins) != 1 or bins[0].name != 'adapter_model.bin':
            raise AssertionError('Pinned PEFT0.5 job expects one explicit adapter_model.bin; ambiguous or changed format rejected')
        tensor_path = bins[0].resolve()
        tok = LlamaTokenizer.from_pretrained(adapter, local_files_only=True)
        tok.pad_token_id, tok.padding_side, tok.model_max_length = 0, 'left', 2048
        audited = LlamaTokenizer.from_pretrained(audit / 'tokenizer_a', local_files_only=True)
        if tok.get_vocab() != audited.get_vocab() or tok.get_added_vocab() != audited.get_added_vocab():
            raise AssertionError('Saved tokenizer or SID token ID ordering differs from audited tokenizer')
        added = tok.get_added_vocab()
        sid_ids = sorted(added.values())
        report['tokenizer_identity'] = {'vocab_size': len(tok), 'added_count': len(added),
             'exact_vocab_and_added_token_ids_match_audit': True, 'added_token_ids': added,
             'special_ids': {'bos': tok.bos_token_id, 'eos': tok.eos_token_id, 'pad': tok.pad_token_id}}
        train_paths = [Path(args.train_dir) / ('rank%d.json' % i) for i in range(world)]
        training = [json.loads(path.read_text()) for path in train_paths]
        if not all(item['status'] == 'PASS' and 3 <= len(item['updates']) <= 9 for item in training):
            raise AssertionError('Reload/generation requires completed real training PASS with at least three bounded updates')
        for item in training:
            if not any(row['nonzero_learning_rate'] for row in item['updates']):
                raise AssertionError('No actual nonzero-LR optimizer boundary was measured')
            changes = item['small_parameter_change_max_abs']
            if not any(value > 0 for name, value in changes.items() if 'lora_' in name):
                raise AssertionError('Training evidence does not show a real LoRA parameter change')
            for family in ['embed_tokens', 'lm_head']:
                if not any(value > 0 for name, value in changes.items() if family in name and 'modules_to_save.default' in name):
                    raise AssertionError('Training evidence does not show changed active SID ' + family)
        report['training_evidence'] = [{'path': str(path), 'sha256': sha(path)} for path in train_paths]
        inventories = [{saved_key(k): v for k, v in item['parameter_inventory'].items() if v['requires_grad']} for item in training]
        if inventories[0].keys() != inventories[1].keys():
            raise AssertionError('Rank training inventories disagree')
        if any(inventories[0][key]['shape'] != inventories[1][key]['shape'] for key in inventories[0]):
            raise AssertionError('Rank training tensor shapes disagree')
        expected = inventories[0]
        if len(expected) != 450 or sum('.lora_' in k for k in expected) != 448:
            raise AssertionError('Unexpected LoRA/SID trained tensor set')
        if any('original_module' in k for k in expected):
            raise AssertionError('Frozen original tensor unexpectedly saved as trained adapter')
        configuration = json.loads((adapter / 'adapter_config.json').read_text())
        if configuration['r'] != 8 or configuration['lora_alpha'] != 16 or configuration['lora_dropout'] != .05:
            raise AssertionError('Saved LoRA hyperparameters differ')
        if set(configuration['modules_to_save']) != {'embed_tokens', 'lm_head'}:
            raise AssertionError('Saved active embedding/head configuration missing')
        if set(configuration['target_modules']) != {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'}:
            raise AssertionError('Saved LoRA target modules differ')
        report['adapter_files'] = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                                   for path in adapter.iterdir() if path.is_file()}
        report['loading_definition'] = {'base': 'original test_ddp.py LoRA branch: full BF16 LLaMA, vanilla HF attention',
            'quantization': 'none during author LoRA inference branch; training was int8',
            'format': tensor_path.name, 'no_new_shims': True,
            'reason': '8-bit Autograd and LoRA-A-only BF16 compatibility target training int8 classes; inference BF16 PEFT Linear uses its own native dtype cast'}
        t = time.monotonic()
        base = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                              device_map={'': local_rank}, local_files_only=True)
        base.resize_token_embeddings(len(tok))
        torch.cuda.synchronize()
        report['base_load_seconds'] = time.monotonic() - t
        guard()
        load_source = inspect.getsource(PeftModel.load_adapter)
        report['peft_load_adapter_identity'] = {'signature': str(inspect.signature(PeftModel.load_adapter)),
            'source_file': inspect.getsourcefile(PeftModel.load_adapter),
            'source_sha256': hashlib.sha256(load_source.encode()).hexdigest()}
        peft_opens = []

        def audit_open(event, event_args):
            if event == 'open' and isinstance(event_args[0], (str, bytes, os.PathLike)):
                if os.fsdecode(event_args[0]) == str(tensor_path):
                    peft_opens.append({'path': str(tensor_path), 'mode': str(event_args[1])})

        sys.addaudithook(audit_open)
        t = time.monotonic()
        model = PeftModel.from_pretrained(base, str(adapter.resolve()), torch_dtype=torch.bfloat16,
                                         device_map={'': local_rank}, is_trainable=False)
        model.eval()
        torch.cuda.synchronize()
        report['adapter_reload_seconds'] = time.monotonic() - t
        report['peft_actual_weight_reads'] = list(peft_opens)
        if not peft_opens:
            raise AssertionError('PEFT did not actually read expected local .bin payload')
        report['base_and_adapter_load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['base_and_adapter_load_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        t = time.monotonic()
        saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
        loaded = get_peft_model_state_dict(model)
        if set(saved) != set(loaded) or set(saved) != set(expected):
            raise AssertionError('Saved/reloaded/trained parameter key sets differ')
        tensor_checks = {}
        for key, value in saved.items():
            actual = loaded[key].detach().cpu()
            if list(value.shape) != expected[key]['shape'] or value.shape != actual.shape:
                raise AssertionError('Tensor shape mismatch: ' + key)
            saved_dtype = expected[key].get('dtype_after_ds')
            if saved_dtype is not None and str(value.dtype) != saved_dtype:
                raise AssertionError('Saved tensor dtype differs from trained tensor: ' + key)
            converted = value.to(actual.dtype)
            if not torch.equal(converted, actual) or not torch.isfinite(actual).all():
                raise AssertionError('Reloaded tensor not exactly saved value in actual inference dtype: ' + key)
            difference = (actual.float() - value.float()).abs()
            tensor_checks[key] = {'shape': list(value.shape), 'saved_dtype': str(value.dtype),
                 'reload_dtype': str(actual.dtype), 'exact_after_explicit_dtype_conversion': True,
                 'max_abs_dtype_conversion_difference': float(difference.max())}
            if key.endswith('embed_tokens.weight') or key.endswith('lm_head.weight'):
                if value.shape != (len(tok), 4096) or not torch.equal(converted[sid_ids], actual[sid_ids]):
                    raise AssertionError('SID embedding/head rows did not strictly restore: ' + key)
                tensor_checks[key]['all_added_sid_rows_equal_after_dtype_conversion'] = True
        if sum(value.numel() for value in saved.values()) != 289652736:
            raise AssertionError('Unexpected adapter trainable element count')
        report['adapter_exact_key_and_value_reload'] = {'keys': len(saved), 'elements': 289652736,
            'all_equal_in_actual_inference_dtype': True, 'full_tensor_checks': tensor_checks,
            'no_missing_keys_tolerated': True, 'verification_seconds': time.monotonic() - t}
        del saved, loaded, expected, inventories, tensor_checks, value, actual, converted, difference
        data_args = SimpleNamespace(dataset='Instruments', data_path=args.data_root, max_his_len=20,
                                    his_sep=', ', index_file='.index.json', add_prefix=False)
        data = SeqRecDataset(data_args, mode='test')
        data.set_prompt(0)
        collate = TestCollator(data_args, tok)
        allowed = data.get_prefix_allowed_tokens_fn(tok)
        matrix = np.load(audit / 'length_rows.npz')
        lengths = matrix['test_input_lengths'][:, 0]
        selection = select_users(lengths, world, np)
        users = list(data.remapped_inters)
        for pairs in selection.values():
            for item in pairs:
                item['user_id'] = users[item['row_idx']]
        report['selection'] = selection
        report['generation_definition'] = {'num_beams': 20, 'num_return_sequences': 20, 'max_new_tokens': 10,
            'prefix_constraint': 'original positional allowed-token callback; no Trie', 'prompt': 0,
            'per_rank_batch': 1, 'distinct_users_per_synchronized_batch': world,
            'warmup': 'q10 users once; excluded from throughput', 'metrics_or_filtering_executed': False,
            'oom_beam_reduction': False, 'allowed_callback_source_sha256': hashlib.sha256(inspect.getsource(data.get_prefix_allowed_tokens_fn).encode()).hexdigest()}
        write(output / ('inference_rank%d.json' % rank), report)
        for index, kind in enumerate(['q10'] + list(selection)):
            guard()
            chosen = selection[kind][rank]
            prep = time.monotonic()
            inputs, _ = collate([data[chosen['row_idx']]])
            if int(inputs['attention_mask'].sum()) != chosen['input_length']:
                raise AssertionError('Real prompt length differs from frozen CPU audit')
            inputs = {key: value.to(local_rank) for key, value in inputs.items()}
            torch.cuda.synchronize()
            preparation_seconds = time.monotonic() - prep
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            with torch.no_grad():
                result = model.generate(**inputs, max_new_tokens=10, prefix_allowed_tokens_fn=allowed,
                    num_beams=20, num_return_sequences=20, output_scores=True,
                    return_dict_in_generate=True, early_stopping=True)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t
            max_seconds = torch.tensor(seconds, device=local_rank)
            dist.all_reduce(max_seconds, op=dist.ReduceOp.MAX)
            suffix = result.sequences[:, inputs['input_ids'].shape[1]:].cpu().tolist()
            if len(suffix) != 20 or not torch.isfinite(result.sequences_scores).all():
                raise AssertionError('Expected twenty finite scored output beams')
            prompt_ids = inputs['input_ids'][0].cpu().tolist()
            eos_positions = []
            for beam in suffix:
                if tok.eos_token_id not in beam:
                    raise AssertionError('Beam did not reach EOS within author max_new_tokens=10')
                eos = beam.index(tok.eos_token_id)
                eos_positions.append(eos)
                for position, token_id in enumerate(beam[:eos + 1]):
                    permitted = allowed(0, torch.tensor(prompt_ids + beam[:position], dtype=torch.long))
                    if permitted is None or token_id not in permitted:
                        raise AssertionError('Generated token violates original positional allowed-token callback')
                if any(token_id not in {tok.eos_token_id, tok.pad_token_id} for token_id in beam[eos + 1:]):
                    raise AssertionError('Nonpadding generated tokens found after EOS')
            row = {'warmup': index == 0, 'selection': kind, **chosen, 'rank_seconds': seconds,
                'max_rank_seconds': float(max_seconds), 'global_users': world,
                'dual_gpu_users_per_second': world / float(max_seconds),
                'cpu_collator_and_transfer_seconds': preparation_seconds,
                'generated_sequence_count': 20, 'generated_effective_tokens': sum(i + 1 for i in eos_positions),
                'output_token_ids': suffix, 'eos_positions': eos_positions, 'eos_all': True,
                'original_positional_callback_verified_all_tokens': True,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            report['timings'].append(row)
            del result, inputs
            write(output / ('inference_rank%d.json' % rank), report)
        report['status'] = 'RELOAD_BEAM20_TECHNICAL_PASS'
    except BaseException as error:
        report['status'] = 'FAIL'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        print(report['error']['traceback'], flush=True)
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        report['cpu_process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and torch.cuda.is_initialized():
            report['cuda_peak_allocated_bytes'] = max([torch.cuda.max_memory_allocated(), report.get('base_and_adapter_load_peak_allocated_bytes', 0)] +
                                                    [row['gpu_peak_allocated_bytes'] for row in report['timings']])
            report['cuda_peak_reserved_bytes'] = max([torch.cuda.max_memory_reserved(), report.get('base_and_adapter_load_peak_reserved_bytes', 0)] +
                                                   [row['gpu_peak_reserved_bytes'] for row in report['timings']])
        write(output / ('inference_rank%d.json' % rank), report)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report['status'] == 'RELOAD_BEAM20_TECHNICAL_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
