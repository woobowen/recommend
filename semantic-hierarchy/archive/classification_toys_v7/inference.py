"""Offline FP16 inference on two GPUs. No automatic experiment execution."""
import json
import os
import subprocess
import time
from pathlib import Path

from common import sha, write_json


def parse_json(text):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON object key: ' + key)
            result[key] = value
        return result
    decoder = json.JSONDecoder(object_pairs_hook=unique_keys)
    for offset, char in enumerate(text):
        if char == '{':
            try:
                result, end = decoder.raw_decode(text[offset:])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    raise ValueError('No complete JSON object in model output')


class Engine:
    def __init__(self, cfg, model_dir, gpus):
        if len(gpus) != 2 or len(set(gpus)) != 2:
            raise ValueError('Exactly two different physical GPU indices required')
        os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        # Never claim GPUs already used by a training/inference compute process.
        listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader,nounits'], text=True)
        physical = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                            '--format=csv,noheader,nounits'], text=True)
        uuids = {int(row.split(',')[0].strip()): row.split(',')[1].strip()
                 for row in physical.strip().splitlines()}
        for gpu in gpus:
            if uuids[gpu] in listing:
                raise RuntimeError(f'GPU {gpu} has a compute process; choose a free pair')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        torch.manual_seed(cfg['seed'])
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Expected exactly two visible CUDA devices')
        for i in range(2):
            if torch.cuda.mem_get_info(i)[0] < 10 * 1024**3:
                raise RuntimeError('Less than 10 GiB free on selected GPU')
        self.cfg = cfg
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        started = time.monotonic()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map='balanced', max_memory={0: '9GiB', 1: '9GiB'},
            attn_implementation='sdpa').eval()
        if any(str(device) in ('cpu', 'disk') for device in self.model.hf_device_map.values()):
            raise RuntimeError('CPU/disk offload is not part of the FP16 protocol')
        self.device = self.model.get_input_embeddings().weight.device
        self.load_seconds = time.monotonic() - started
        self.calls = 0
        self.enforcer_data = None

    def encode(self, system, user):
        return self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            add_generation_prompt=True, tokenize=True, return_tensors='pt')

    def fits(self, system, user, max_new):
        return self.encode(system, user).shape[-1] + max_new + 32 <= self.cfg['context_tokens']

    def generate(self, system, user, max_new, log_path, schema=None):
        tokens = self.encode(system, user)
        if tokens.shape[-1] + max_new + 32 > self.cfg['context_tokens']:
            raise ValueError('Prompt exceeds the frozen token budget; do not truncate item text silently')
        tokens = tokens.to(self.device)
        constraints = {}
        if schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
            if self.enforcer_data is None:
                self.enforcer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
            constraints['prefix_allowed_tokens_fn'] = build_transformers_prefix_allowed_tokens_fn(
                self.enforcer_data, JsonSchemaParser(schema))
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=tokens,
                attention_mask=self.torch.ones_like(tokens), do_sample=False, temperature=1.0, top_p=1.0,
                max_new_tokens=max_new, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
                return_dict_in_generate=False, **constraints)
        generated = output[0, tokens.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.calls += 1
        write_json(log_path, {'system': system, 'user': user, 'response': text,
            'input_tokens': tokens.shape[-1], 'output_tokens': generated.numel(),
            'seconds': time.monotonic()-started, 'max_new_tokens': max_new,
            'possibly_truncated': generated.numel() >= max_new})
        if schema is not None:
            write_json(Path(log_path).with_suffix('.schema.json'), schema)
        if generated.numel() >= max_new:
            raise ValueError('Generation hit its token limit; raw output retained')
        return parse_json(text)


def smoke():
    import argparse
    from common import roots
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1')
    args = ap.parse_args()
    cfg, root, prep, out = roots()
    assert (prep / 'model_download_verified.json').exists(), 'Finish and verify download first'
    engine = Engine(cfg, root / cfg['model'], [int(x) for x in args.gpus.split(',')])
    prompt = ('This is a software installation test with synthetic products. '
              'Classify A: red lipstick, B: wooden building blocks. '
              'Return only JSON with keys A and B; values must be cosmetics and toys respectively.')
    result = engine.generate('Follow the requested JSON schema exactly.', prompt, 128,
                             prep / 'synthetic_smoke_raw.json')
    assert result == {'A': 'cosmetics', 'B': 'toys'}, result
    from prompts import SYSTEM, CLASSIFY
    synthetic_taxonomy = {'categories': [
        {'id': 'F001', 'name': 'Lipstick', 'coarse_id': 'C01', 'definition': 'Lip color cosmetics.'},
        {'id': 'F002', 'name': 'Building blocks', 'coarse_id': 'C02', 'definition': 'Toy blocks for construction.'}],
        'unknown': {'id': 'UNKNOWN', 'definition': 'Insufficient product information.'}}
    products = [{'item_id': 1, 'text': 'A red lipstick for lip makeup.'},
                {'item_id': 2, 'text': 'Wooden toy building blocks for children.'}]
    exact_prompt = CLASSIFY.replace('{taxonomy}', json.dumps(synthetic_taxonomy)).replace('{products}', json.dumps(products))
    structured = engine.generate(SYSTEM, exact_prompt, 256, prep / 'classifier_schema_smoke_raw.json')
    assert {r['item_id']: r['category_id'] for r in structured['items']} == {1: 'F001', 2: 'F002'}
    assert all(type(r['needs_review']) is bool for r in structured['items'])
    # Exercise the planned maximum context/cache budget without real item labels.
    filler = 'This neutral sentence is padding for a software memory test. '
    user = ' '.join([filler] * 700) + '\nReturn only {"ok":true}.'
    while engine.encode('Follow instructions and output JSON.', user).shape[-1] > cfg['context_tokens'] - 160:
        user = user[len(filler):]
    memory_result = engine.generate('Follow instructions and output JSON.', user, 96,
                                   prep / 'context_smoke_raw.json')
    assert memory_result == {'ok': True}, memory_result
    import transformers, accelerate
    write_json(prep / 'model_smoke_verified.json', {
        'passed': True, 'synthetic_only': True, 'real_items_classified': 0,
        'gpus': args.gpus, 'dtype': str(engine.model.dtype),
        'device_map': engine.model.hf_device_map, 'load_seconds': engine.load_seconds,
        'context_tokens': cfg['context_tokens'], 'torch': engine.torch.__version__,
        'transformers': transformers.__version__, 'accelerate': accelerate.__version__,
        'peak_allocated_GiB': [engine.torch.cuda.max_memory_allocated(i)/1024**3 for i in range(2)],
        'peak_reserved_GiB': [engine.torch.cuda.max_memory_reserved(i)/1024**3 for i in range(2)],
        'config_sha256': sha(Path(__file__).with_name('config.json'))})
    print('synthetic_model_smoke_passed', flush=True)


if __name__ == '__main__':
    smoke()
