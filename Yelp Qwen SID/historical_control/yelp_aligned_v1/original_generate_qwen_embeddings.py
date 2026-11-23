"""Generate new item vectors with local Qwen weights and WMH mean pooling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['beauty', 'sports', 'toys'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'models/Qwen2.5-7B-Instruct')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh run directory')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, TF_CPP_MIN_LOG_LEVEL='3',
                      TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from src.models.components.network_blocks.embedding_aggregator import EmbeddingAggregator
    from src.models.components.network_blocks.aggregation_strategy import MeanAggregation

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if cfg.model_type != 'qwen2':
        raise ValueError('Expected the original Qwen2.5 model')
    files = sorted((ROOT/'original_assets/data/amazon_data'/args.dataset/'items').glob('*.tfrecord.gz'))
    if not files:
        raise FileNotFoundError('No item TFRecords found')
    rows = {}
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    for f in files:
        for raw in tf.data.TFRecordDataset(str(f), compression_type='GZIP').with_options(options):
            fields = tf.train.Example.FromString(raw.numpy()).features.feature
            idx = int(fields['id'].int64_list.value[0])
            text = fields['text'].bytes_list.value
            if idx in rows or len(text) != 1:
                raise ValueError(f'Duplicate ID or unexpected item text: {idx}')
            rows[idx] = text[0].decode('utf-8')
    if sorted(rows) != list(range(len(rows))):
        raise ValueError('Item IDs must be contiguous from zero for the WMH SID map')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Use at least two free 11-GiB GPUs for the full FP16 Qwen model')
    for i in range(torch.cuda.device_count()):
        if torch.cuda.mem_get_info(i)[0] < 9*1024**3:
            raise RuntimeError(f'Selected GPU {i} has less than 9 GiB free; select free GPUs')
    max_memory = {i: '9GiB' for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '32GiB'
    model = AutoModel.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float16, device_map='auto', max_memory=max_memory,
        low_cpu_mem_usage=True).eval()
    model.config.use_cache = False
    aggregate = EmbeddingAggregator(MeanAggregation())
    device = model.get_input_embeddings().weight.device
    output = torch.empty(len(rows), cfg.hidden_size, dtype=torch.float32)
    start = time.time()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            texts = [rows[i] for i in range(offset, min(offset+args.batch_size, len(rows)))]
            tokens = tokenizer(texts, max_length=128, padding='max_length', truncation=True,
                               add_special_tokens=True, return_tensors='pt').to(device)
            hidden = model(**tokens, use_cache=False).last_hidden_state
            vectors = aggregate(hidden, tokens.attention_mask.to(hidden.device)).float().cpu()
            if not torch.isfinite(vectors).all():
                raise ValueError('Non-finite Qwen item vector')
            output[offset:offset+len(texts)] = vectors
            if offset % 100 == 0:
                print(f'{offset}/{len(rows)} items, elapsed={time.time()-start:.0f}s', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    torch.save(output, temporary)
    temporary.replace(args.output)
    metadata = {'dataset': args.dataset, 'model': str(args.model), 'architecture': cfg.model_type,
        'shape': list(output.shape), 'pooling': 'WMH MeanAggregation, right padding',
        'max_length': 128, 'weight_dtype': 'float16', 'stored_dtype': 'float32',
        'device_map': model.hf_device_map, 'source': 'TFRecord text (existing embedding field ignored)',
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
