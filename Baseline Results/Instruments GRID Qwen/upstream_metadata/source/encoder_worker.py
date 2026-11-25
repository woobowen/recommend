"""Text-only worker. Never opens SID tables or uses SID-dependent sampling."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', USE_TF='0',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
import peft.tuners.lora.model as lora_dispatch
from llm2vec import LLM2Vec
from llm2vec.models import LlamaBiModel
from runtime_common import ROOT, read_json, read_jsonl, sha, status, write_json


def load_encoder(cfg):
    base = ROOT / 'models/Meta-Llama-3-8B-Instruct'
    mntp = ROOT / 'models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'
    supervised = ROOT / 'models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    config.use_cache = False
    mapping = {'embed_tokens': 0, 'rotary_emb': 0, 'norm': 1}
    mapping.update({f'layers.{i}': 0 if i < 16 else 1 for i in range(32)})
    model = LlamaBiModel.from_pretrained(
        base, config=config, torch_dtype=torch.float16, device_map=mapping,
        attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)
    assert not getattr(model, 'is_loaded_in_4bit', False)
    assert not getattr(model, 'is_loaded_in_8bit', False)
    lora_dispatch.is_bnb_available = lambda: False
    lora_dispatch.is_bnb_4bit_available = lambda: False
    model = PeftModel.from_pretrained(model, mntp, local_files_only=True,
                                      autocast_adapter_dtype=False).merge_and_unload()
    model = PeftModel.from_pretrained(model, supervised, local_files_only=True,
                                      autocast_adapter_dtype=False)
    model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model.config.use_cache = False
    assert all(layer.self_attn.is_causal is False for layer in model.get_base_model().layers)
    tokenizer = AutoTokenizer.from_pretrained(mntp, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return LLM2Vec(model, tokenizer, pooling_mode='mean',
                   max_length=cfg['max_length'], doc_max_length=cfg['doc_max_length'],
                   skip_instruction=True).eval()


def encode_variant(encoder, spec, cfg, out, state_path, dataset):
    out.mkdir(parents=True, exist_ok=True)
    text_path = Path(spec['path'])
    assert sha(text_path) == spec['sha256'], 'Input text changed'
    rows = read_jsonl(text_path)
    assert len(rows) == cfg['expected_items'][dataset]
    assert [row['item_id'] for row in rows] == list(range(len(rows)))
    assert all(set(row) == {'item_id', 'text'} and row['text'].strip() for row in rows)
    signature = {'text_sha256': spec['sha256'], 'encoding_config': cfg,
                 'encoder_source_sha256': sha(Path(__file__))}
    done_path = out / 'encoding_complete.json'
    if done_path.exists():
        previous = read_json(done_path)
        assert previous['signature'] == signature
        assert previous['embedding_sha256'] == sha(out / 'embeddings_normalized.npy')
        status(state_path, 'variant_already_complete', dataset=dataset, variant=spec['name'])
        return
    # The ordering depends only on text length and item_id, never on codes/categories.
    order = sorted(range(len(rows)), key=lambda i: (-len(rows[i]['text']), i))
    token_lengths = [len(encoder.tokenizer(row['text'], add_special_tokens=False)['input_ids'])
                     for row in rows]
    checkpoint_path = out / 'encoding_checkpoint.json'
    shape = (len(rows), cfg['output_dimension'])
    if checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        assert checkpoint['signature'] == signature
        cursor = checkpoint['completed_order_rows']
        vectors = np.lib.format.open_memmap(out / 'embeddings_normalized.npy', mode='r+')
        assert vectors.shape == shape
        before_norm = np.lib.format.open_memmap(out / 'norms_before_normalization.npy', mode='r+')
    else:
        cursor = 0
        vectors = np.lib.format.open_memmap(out / 'embeddings_normalized.npy', mode='w+',
                                            dtype=np.float32, shape=shape)
        before_norm = np.lib.format.open_memmap(out / 'norms_before_normalization.npy', mode='w+',
                                                dtype=np.float32, shape=(len(rows),))
        write_json(checkpoint_path, {'signature': signature, 'completed_order_rows': 0})
    np.save(out / 'item_ids.npy', np.arange(len(rows), dtype=np.int64))
    started = time.monotonic()
    start_cursor = cursor
    oom_splits = 0
    truncated_input_count = 0

    def batch(ids):
        nonlocal oom_splits, truncated_input_count
        prepared = [encoder.prepare_for_tokenization(
            encoder._convert_to_str(cfg['instruction'], rows[i]['text'])) for i in ids]
        features = encoder.tokenize(prepared)
        try:
            features = {key: value.to('cuda:0') for key, value in features.items()}
            with torch.inference_mode():
                raw = encoder(features).float().cpu()
        except torch.cuda.OutOfMemoryError:
            if len(ids) == 1:
                raise
            del features
            torch.cuda.empty_cache()
            oom_splits += 1
            split = len(ids) // 2
            left = batch(ids[:split]); right = batch(ids[split:])
            return torch.cat([left[0], right[0]]), torch.cat([left[1], right[1]])
        assert raw.shape == (len(ids), cfg['output_dimension'])
        assert torch.isfinite(raw).all()
        norms = raw.norm(dim=-1)
        assert (norms > 1e-12).all()
        return torch.nn.functional.normalize(raw, p=2, dim=-1), norms

    while cursor < len(order):
        block_end = min(len(order), cursor + 128)
        for start in range(cursor, block_end, cfg['batch_size']):
            ids = order[start:min(block_end, start + cfg['batch_size'])]
            normalized, norms = batch(ids)
            vectors[ids] = normalized.numpy()
            before_norm[ids] = norms.numpy()
        cursor = block_end
        vectors.flush(); before_norm.flush()
        elapsed = time.monotonic() - started
        processed = cursor - start_cursor
        rate = processed / elapsed
        write_json(checkpoint_path, {'signature': signature, 'completed_order_rows': cursor})
        status(state_path, 'encoding', dataset=dataset, variant=spec['name'],
               completed=cursor, total=len(order), items_per_second=rate,
               estimated_remaining_seconds=(len(order) - cursor) / rate,
               physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'])
    norms_after = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
    assert np.isfinite(vectors).all() and np.max(np.abs(norms_after - 1)) < 1e-5
    report = {'complete': True, 'signature': signature, 'items': len(rows),
              'shape': list(shape), 'dtype': str(vectors.dtype),
              'embedding_sha256': sha(out / 'embeddings_normalized.npy'),
              'item_ids_sha256': sha(out / 'item_ids.npy'),
              'norm_min': float(norms_after.min()), 'norm_max': float(norms_after.max()),
              'pre_norm_min': float(before_norm.min()), 'pre_norm_max': float(before_norm.max()),
              'title_tokens_over_doc_limit': sum(n > cfg['doc_max_length'] for n in token_lengths),
              'token_length_p50_p95_max': np.quantile(token_lengths, [.5, .95, 1]).tolist(),
              'oom_batch_splits': oom_splits,
              'physical_gpus': os.environ['CUDA_VISIBLE_DEVICES'],
              'gpu_peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)],
              'read_inputs': [str(text_path)],
              'semantic_vector_quantization': False,
              'finished_unix': time.time()}
    write_json(done_path, report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--dataset', choices=['instrument'], required=True)
    args = ap.parse_args()
    run = Path(args.run)
    cfg = read_json(run / 'encoding_config.json')
    state_path = run / args.dataset / 'encoding_status.json'
    lock = (run / f'{args.dataset}.encoding.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_locks = []
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices) == 2 and len(set(devices)) == 2
    for gpu in sorted(devices, key=int):
        gpu_lock = (ROOT / 'stage_gpu_locks' / f'gpu_{int(gpu)}.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_locks.append(gpu_lock)
    assert torch.cuda.device_count() == 2
    for i in range(2):
        assert torch.cuda.mem_get_info(i)[0] > 9 * 1024**3
    try:
        status(state_path, 'loading_model', dataset=args.dataset,
               physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'])
        torch.set_num_threads(2)
        torch.manual_seed(cfg['seed'])
        encoder = load_encoder(cfg)
        for spec in cfg['inputs'][args.dataset]:
            encode_variant(encoder, spec, cfg, run / args.dataset / spec['name'], state_path, args.dataset)
        status(state_path, 'complete', dataset=args.dataset,
               physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'])
    except Exception as exc:
        status(state_path, 'failed', dataset=args.dataset,
               error_type=type(exc).__name__, error=str(exc)[:1000])
        raise


if __name__ == '__main__':
    main()
