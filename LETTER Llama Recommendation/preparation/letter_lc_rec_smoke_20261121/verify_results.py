"""Audit completed smoke artifacts and write a compact reviewable summary."""
from pathlib import Path
import datetime
import hashlib
import json
import re
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
queue=json.loads((HERE/'latest_queue.json').read_text())
run=Path(queue['launch'])
state=json.loads((run/'status.json').read_text())
assert state['stage']=='complete' and state['complete']
assets=json.loads((HERE/'assets_audit.json').read_text())
assert len(assets['official_files'])==57
assert all(hashlib.sha256((HERE/'vendor/LETTER'/p).read_bytes()).hexdigest()==h
           for p,h in assets['official_files'].items())
download=json.loads((HERE/'model_download_verified.json').read_text())
assert download['complete']
summary={'completed_beijing':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'run':str(run),'source_commit':assets['source_commit'],'model_id':assets['model_id'],
    'model_revision':assets['model_revision'],'official_files':57,'model_bytes':download['total_bytes'],
    'model_files':len(download['files']),'formal_training_started':False,'datasets':{}}
for ds in ['Instruments','Beauty']:
    path=run/ds
    result=json.loads((path/'result.json').read_text())
    config=json.loads((path/'config.json').read_text())
    assert result['complete'] and result['optimizer_steps']==6
    assert result['reload_max_logit_error']==0.0 and result['sid_embedding_max_change']>0
    events=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
    steps=[e for e in events if e['stage']=='optimizer_step']
    assert len(steps)==6 and set(e['task'] for e in steps)==set(config['tasks'])
    assert all(e['gradient_norms'][k]>0 for e in steps for k in ['lora','embedding','head'])
    targets=None;mode_results={}
    for mode in ['unshuffle','shuffle_3','shuffle_4']:
        data=json.loads((path/f'test_{mode}.json').read_text())
        assert data['users']==4 and data['beam']==20
        assert data['adapter']==str(path/'adapter')
        current=[(r['user_row'],r['target']) for r in data['rows']]
        if targets is None:targets=current
        assert targets==current
        for r in data['rows']:
            assert len(r['candidates'])==20
            assert all(re.fullmatch(r'<a_\d+><b_\d+><c_\d+><d_\d+>',s) for s in r['candidates'])
            if mode=='unshuffle':assert r['changed_history_items']==0
        if mode!='unshuffle':assert sum(r['changed_history_items'] for r in data['rows'])>0
        recalculated={k:sum(r['metrics'][k] for r in data['rows'])/4 for k in data['metrics']}
        assert recalculated==data['metrics']==result['tests'][mode]
        mode_results[mode]={'passed':True,'users':4,'beam':20,'metrics':data['metrics'],
            'valid_catalog_candidates':sum(r['valid_catalog_candidates'] for r in data['rows']),
            'total_candidates':80}
    fp16_modes={}
    for mode in ['unshuffle','shuffle_3','shuffle_4']:
        f=json.loads((path/f'test_fp16_{mode}.json').read_text())
        assert f['passed'] and f['users']==4 and f['beam']==20 and f['precision']=='unquantized_fp16'
        assert targets==[(r['user_row'],r['target']) for r in f['rows']]
        assert all(len(r['candidates'])==20 and all(re.fullmatch(r'<a_\d+><b_\d+><c_\d+><d_\d+>',s)
            for s in r['candidates']) for r in f['rows'])
        fp16_modes[mode]={'passed':True,'users':4,'beam':20,'metrics':f['metrics']}
    summary['datasets'][ds]={**assets['datasets'][ds], 'tasks':config['tasks'],
        'temperature':config['temperature'],'optimizer_steps':6,
        'validation_loss':result['validation_loss'],'reload_max_logit_error':0.0,
        'max_train_allocated_mib':[max(s['max_allocated_mib'][i] for s in steps) for i in [0,1]],
        'tests':mode_results,'fp16_tests':fp16_modes}
fp16=json.loads((HERE/'fp16_inference_verified.json').read_text())
assert fp16['complete'] and all(r['passed'] for r in fp16['datasets'].values())
summary['unquantized_fp16_inference']=fp16
(HERE/'verified_smoke_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# LETTER-LC-Rec 小规模测试','',
    '完整第一代 LLaMA-7B 基础模型，Int8 主干、FP16 计算、LoRA；每数据集 6 次参数更新、每种推理条件 4 位用户、beam20。',
    '两组均完成训练、验证、adapter 保存重载以及 unshuffle/shuffle_3/shuffle_4 生成和指标计算。重载前后 logits 最大差值均为 0。',
    '', '| 数据集 | 验证 loss | GPU0 训练峰值分配 MiB | GPU1 训练峰值分配 MiB | 重载 logits 最大差值 |',
    '|---|---:|---:|---:|---:|']
for ds,r in summary['datasets'].items():
    lines.append(f"| {ds} | {r['validation_loss']:.6f} | {r['max_train_allocated_mib'][0]:.1f} | {r['max_train_allocated_mib'][1]:.1f} | 0 |")
lines.extend(['','| 数据集 | 模式 | 用户数 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |',
              '|---|---|---:|---:|---:|---:|---:|'])
for ds,r in summary['datasets'].items():
    for mode,m in r['tests'].items():
        v=m['metrics']
        lines.append(f"| {ds} | {mode} | 4 | {v['hit@5']:.6f} | {v['hit@10']:.6f} | {v['ndcg@5']:.6f} | {v['ndcg@10']:.6f} |")
lines.extend(['','上表为 Int8 推理。另已补测作者测试脚本使用的未量化主干：在 2080 Ti 上改用 FP16，两数据集的三种顺序、每模式 4 用户、beam20 均通过，命中和 NDCG 也均为 0。',
              '当前命中指标均为 0。这是仅训练 6 次更新、测试 4 位用户的运行检查，不能用于评价模型推荐效果或比较 shuffle 的性能差异。',
              '该短测未启动全量正式训练。原作者实际采用位次允许集；候选不保证属于商品目录。详细配置、硬件适配与原始记录见同目录准备说明及 JSON。'])
(HERE/'小规模测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
