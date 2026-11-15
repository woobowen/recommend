#!/usr/bin/env python3
"""Preflight adapter payload structure and fixed inference contract without model load."""
import hashlib,json,sys
from pathlib import Path
import torch
from formal_evaluate import expected_adapter_shapes
adapter=Path(sys.argv[1]); manifest=Path(sys.argv[2]);weights=adapter/'adapter_model.bin'
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39'
cfg=json.loads((adapter/'adapter_config.json').read_text())
assert (cfg['r'],cfg['lora_alpha'],cfg['lora_dropout'])==(8,16,0.05)
assert set(cfg['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'}
assert set(cfg['modules_to_save'])=={'embed_tokens','lm_head'} and cfg.get('bias','none')=='none'
state=torch.load(weights,map_location='cpu',weights_only=True);expected=expected_adapter_shapes()
assert set(state)==set(expected) and len(state)==450
assert all(tuple(state[k].shape)==shape and torch.isfinite(state[k]).all() for k,shape in expected.items())
print(json.dumps({'status':'PASS','scope':'CPU checkpoint structure and finite weights only, not PEFT reload','keys':len(state),'elements':sum(x.numel() for x in state.values()),'adapter_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
