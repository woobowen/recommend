"""Beauty production provenance; CPU-readable and shared by launch and acceptance."""
import hashlib
import json
import math
from pathlib import Path
from formal_control import durable_write
from formal_train import checkpoint_required_files, verify_checkpoint
from offline_metrics import ASSETS

STEPS = (2053, 4107, 6160, 8214)
AXIS = '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
SYNTHETIC = 'SYNTHETIC_CPU_INTEGRATION_ONLY'


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular artifact required: '+str(path))
    return {'path':str(path.resolve()), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def check_file(row, path=None):
    actual = identity(path or row['path'])
    need(all(actual[k] == row[k] for k in ('bytes','sha256')), 'artifact hash mismatch: '+actual['path'])
    return actual


def scope_guard(path, scope, record=None):
    need(scope in ('FORMAL', SYNTHETIC), 'unknown evidence scope')
    if scope == SYNTHETIC:
        need(SYNTHETIC in str(path), 'test path must be isolated and labeled')
        if record is not None:
            need(record.get('scope') == SYNTHETIC, 'synthetic evidence scope missing')
    else:
        need('SYNTHETIC' not in str(path) and 'preflight' not in str(path).lower(), 'nonformal path')
        if record is not None:
            need(record.get('scope', 'FORMAL') == 'FORMAL', 'nonformal evidence')


def verify_training(root, commit, config_sha, scope='FORMAL'):
    """Prove native earliest-minimum selection, retained entities and export tensors."""
    root = Path(root); scope_guard(root, scope)
    events=[]
    for rank in (0,1):
        report=load(root/f'rank{rank}.json');scope_guard(root,scope,report)
        need(report['status']=='TRAINING_COMPLETE' and report['rank']==rank and not report['probe'], 'training incomplete/probe')
        need(report['run_commit']==commit and report['config_sha256']==config_sha, 'training RUN/config mismatch')
        need(report['global_step']==8214 and report['epoch']==4.0, 'formal epoch/update count')
        need(report['data']['train_rows']==131413 and report['data']['valid_rows']==44726 and
             not report['data']['validation_subset_preflight_only'] and report['data']['tokenizer_vocab']==32965, 'full training/validation scope')
        need({k:v['sha256'] for k,v in report['data_identity'].items()}==ASSETS, 'training data identity')
        ev=[json.loads(x) for x in (root/f'events.rank{rank}.jsonl').read_text().splitlines()]
        validation=[x for x in ev if x['event']=='log' and 'eval_loss' in x.get('values',{})]
        need([(x['step'],x['epoch']) for x in validation]==list(zip(STEPS,(1.,2.,3.,4.))), 'four ordered complete validation events required')
        need(all(math.isfinite(x['values']['eval_loss']) for x in validation), 'nonfinite validation loss')
        events.append((ev,validation))
    losses=[x['values']['eval_loss'] for x in events[0][1]]
    need(losses==[x['values']['eval_loss'] for x in events[1][1]], 'rank validation disagreement')
    receipts={}
    for p in sorted((root/'checkpoint_receipts').glob('*.json')):
        r=load(p);step=r['global_step'];need(step in STEPS, 'unexpected checkpoint receipt')
        need(r['run_commit']==commit and r['config_sha256']==config_sha, 'receipt source mismatch')
        need({k:v['sha256'] for k,v in r['data_identity'].items()}==ASSETS, 'receipt data mismatch')
        m=r['manifest'];raw=r['manifest_json'].encode('utf-8')
        need(json.loads(raw)==m,'receipt raw manifest content')
        need(hashlib.sha256(raw).hexdigest()==r['manifest_sha256'], 'receipt manifest hash')
        need(m['collective_ranks']==2 and m['global_step']==step and m['run_commit']==commit and m['config_sha256']==config_sha, 'collective completion identity')
        rows=m['files'];need(len(rows)==len({x['path'] for x in rows}), 'duplicate checkpoint member')
        need(checkpoint_required_files(step)<={x['path'] for x in rows}, 'incomplete checkpoint manifest')
        need(all(x['bytes']>0 and len(x['sha256'])==64 for x in rows), 'invalid checkpoint member identity')
        need(r['epoch']==STEPS.index(step)+1 and r['eval_loss']==losses[STEPS.index(step)], 'receipt validation binding')
        need(Path(r['checkpoint_path'])==root/f'checkpoint-{step}', 'foreign checkpoint path')
        if step in receipts:need(receipts[step]['manifest_sha256']==r['manifest_sha256'] and receipts[step]['eval_loss']==r['eval_loss'], 'conflicting receipt attempts')
        receipts[step]=r
    need(set(receipts)==set(STEPS), 'missing epoch receipt')
    # Transformers 4.31 uses a strict comparator: an equal loss keeps the earlier best.
    best_index=min(range(4),key=lambda i:losses[i]);best_step=STEPS[best_index]
    selected=load(root/'best_selection.json');best=root/f'checkpoint-{best_step}'
    need(selected['run_commit']==commit and Path(selected['checkpoint'])==best and selected['value']==losses[best_index], 'best selection mismatch')
    need(selected['metric']=='eval_loss' and selected['greater_is_better'] is False and selected['completed_training_step']==8214 and selected['loaded_best_checkpoint_step']==best_step, 'best restoration identity')
    kept=list(root.glob('checkpoint-*'));need(len(kept)==2 and best in kept and root/'checkpoint-8214' in kept, 'best/latest retention policy')
    for cp in set((best,root/'checkpoint-8214')):
        verify_checkpoint(cp,commit,config_sha)
        need(sha(cp/'complete.json')==receipts[int(cp.name.split('-')[-1])]['manifest_sha256'], 'retained manifest differs from receipt')
    for ev,_ in events:
        restored=[x for x in ev if x['event']=='strict_ds_restored']
        need(restored and Path(restored[-1]['path'])==best and restored[-1]['global_step']==best_step, 'strict best restore proof missing')
    adapter=root/'best_adapter'
    need(sha(adapter/'adapter_model.bin')==selected['adapter_sha256'], 'exported Adapter SHA')
    import torch
    checkpoint=torch.load(best/'adapter_model.bin',map_location='cpu',weights_only=False)
    exported=torch.load(adapter/'adapter_model.bin',map_location='cpu',weights_only=False)
    need(set(checkpoint)==set(exported) and all(torch.equal(v,exported[k]) for k,v in checkpoint.items()), 'export is not selected checkpoint tensor state')
    if scope=='FORMAL':
        from formal_evaluate import expected_adapter_shapes
        need({k:tuple(v.shape) for k,v in exported.items()}==expected_adapter_shapes(), 'Adapter/Embedding/Head shapes')
        cfg=load(adapter/'adapter_config.json')
        need((cfg['r'],cfg['lora_alpha'],cfg['lora_dropout'])==(8,16,.05) and
             set(cfg['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'} and
             set(cfg['modules_to_save'])=={'embed_tokens','lm_head'}, 'frozen exported LoRA configuration')
    token_names=('tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json')
    for name in token_names:need(sha(best/name)==sha(adapter/name), 'Tokenizer differs from selected checkpoint: '+name)
    files={p.name:{k:v for k,v in identity(p).items() if k!='path'} for p in sorted(adapter.iterdir()) if p.is_file()}
    return {'verified':True,'scope':scope,'run_commit':commit,'config_sha256':config_sha,'best_step':best_step,
            'adapter_files':files,'selection':identity(root/'best_selection.json'),
            'receipts':[identity(p) for p in sorted((root/'checkpoint_receipts').glob('*.json'))]}


def verify_predictions(test, commit, best, condition, scope='FORMAL'):
    test=Path(test);scope_guard(test,scope);need(best['scope']==scope, 'model evidence scope mismatch')
    identities=[]
    for rank in (0,1):
        r=load(test/f'rank{rank}.json');scope_guard(test,scope,r)
        need(r['status']=='PASS' and r['rank']==rank and r['preflight_only'] is False and r['RUN_COMMIT']==commit, 'prediction completion/RUN')
        need(r['completed_users']==(11182 if rank==0 else 11181), 'prediction user count')
        need({k:v['sha256'] for k,v in r['data_identity'].items()}==ASSETS, 'prediction data mismatch')
        need(r['adapter_files']==best['adapter_files'], 'prediction is not selected Adapter/Tokenizer')
        if condition!='normal':
            need(r['shuffle']['n']==int(condition[-1]) and r['shuffle']['protocol']=='GRID-LETTER-SHUFFLE-V1', 'Shuffle condition mismatch')
        identities.append(identity(test/f'predictions.rank{rank}.jsonl'))
    return {'verified':True,'scope':scope,'prediction_run_commit':commit,'prediction_files':identities,'adapter_files':best['adapter_files']}
