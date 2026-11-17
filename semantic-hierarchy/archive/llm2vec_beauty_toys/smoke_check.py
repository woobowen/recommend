"""Offline two-GPU compatibility check, using eight fixed synthetic products only.

Not a full experiment. Call after all files are SHA256 verified, with
CUDA_VISIBLE_DEVICES=0,1. Avoid LLM2Vec.encode(), which in version 0.2.3
replicates the model across visible GPUs rather than respecting sharding.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', TF_CPP_MIN_LOG_LEVEL='3')
import json
import fcntl
from pathlib import Path
import time
import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
import peft.tuners.lora.model as lora_dispatch
from llm2vec import LLM2Vec
from llm2vec.models import LlamaBiModel

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE = ROOT / 'codex/llm2vec_sid_hierarchy_20261113'
BASE = ROOT / 'models/Meta-Llama-3-8B-Instruct'
MNTP = ROOT / 'models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'
SUP = ROOT / 'models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'
assert json.loads((HERE / 'preparation/model_download_verified.json').read_text())['complete']
physical_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
assert len(physical_gpus) == 2 and len(set(physical_gpus)) == 2
gpu_locks = []
for gpu in sorted(physical_gpus, key=int):
    lock = (ROOT / 'stage_gpu_locks' / f'gpu_{int(gpu)}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_locks.append(lock)
assert torch.cuda.device_count() == 2, 'Expose precisely the two reserved GPUs'
for i in range(2):
    free, total = torch.cuda.mem_get_info(i)
    assert free > 9 * 1024**3, f'GPU {i} has insufficient free memory'

started = time.monotonic()
config = AutoConfig.from_pretrained(BASE, local_files_only=True)
config.use_cache = False
device_map = {'embed_tokens': 0, 'rotary_emb': 0, 'norm': 1}
device_map.update({f'layers.{i}': 0 if i < 16 else 1 for i in range(32)})
model = LlamaBiModel.from_pretrained(
    BASE, config=config, torch_dtype=torch.float16, device_map=device_map,
    attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)
# The shared parent environment contains bitsandbytes, whose optional Triton
# import needs a system compiler. This model uses ordinary FP16 Linear layers;
# disable only PEFT's unused 4/8-bit dispatch, without changing shared packages.
assert not getattr(model, 'is_loaded_in_4bit', False)
assert not getattr(model, 'is_loaded_in_8bit', False)
lora_dispatch.is_bnb_available = lambda: False
lora_dispatch.is_bnb_4bit_available = lambda: False
model = PeftModel.from_pretrained(model, MNTP, local_files_only=True,
                                  autocast_adapter_dtype=False)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, SUP, local_files_only=True,
                                  autocast_adapter_dtype=False)
model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(MNTP, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
encoder = LLM2Vec(model, tokenizer, pooling_mode='mean', max_length=256,
                  doc_max_length=224, skip_instruction=True).eval()
load_seconds = time.monotonic() - started

texts = ['Title: Moisturizing red lipstick', 'Title: Hydrating red lip color',
         'Title: Red toy racing car', 'Title: Wooden building blocks for children',
         'Title: Lavender body lotion', 'Title: Unscented moisturizing body cream',
         'Title: A set of miniature toy cars', 'Title: Gentle foaming facial cleanser']
instruction = 'Represent this product for semantic similarity:'


def encode_batch(batch):
    prepared = [encoder.prepare_for_tokenization(encoder._convert_to_str(instruction, text))
                for text in batch]
    features = encoder.tokenize(prepared)
    features = {key: value.to('cuda:0') for key, value in features.items()}
    with torch.inference_mode():
        vectors = encoder(features).float().cpu()
    assert vectors.shape == (len(batch), 4096)
    assert torch.isfinite(vectors).all() and (vectors.norm(dim=-1) > 0).all()
    return torch.nn.functional.normalize(vectors, dim=-1)


encode_batch(texts[:2])  # warm-up
for i in range(2):
    torch.cuda.synchronize(i)
t = time.monotonic()
vectors = torch.cat([encode_batch(texts[i:i+2]) for i in range(0, len(texts), 2)])
for i in range(2):
    torch.cuda.synchronize(i)
elapsed = time.monotonic() - t
single = encode_batch(texts[:1])
invariance = float(torch.nn.functional.cosine_similarity(single, vectors[:1]).item())
assert invariance > 0.999, f'Batch/padding dependence is excessive: {invariance}'
assert (vectors.norm(dim=-1) - 1).abs().max() < 1e-5
attention_flags = [getattr(layer.self_attn, 'is_causal', None)
                   for layer in model.get_base_model().layers]
assert all(flag is False for flag in attention_flags), 'Expected bidirectional attention'
report = {'complete': True, 'full_experiment_started': False,
          'optional_bitsandbytes_dispatch_disabled': True,
          'load_seconds': load_seconds, 'synthetic_items': len(texts),
          'encoding_seconds': elapsed, 'synthetic_items_per_second': len(texts)/elapsed,
          'batch_invariance_cosine': invariance, 'attention_is_causal_flags': attention_flags,
          'all_attention_modules_noncausal': all(flag is False for flag in attention_flags),
          'shape': list(vectors.shape), 'dtype': str(vectors.dtype),
          'norm_min': vectors.norm(dim=-1).min().item(),
          'norm_max': vectors.norm(dim=-1).max().item(),
          'gpu_peak_allocated_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)],
          'gpu_names': [torch.cuda.get_device_name(i) for i in range(2)]}
(HERE / 'preparation/smoke_check_verified.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
