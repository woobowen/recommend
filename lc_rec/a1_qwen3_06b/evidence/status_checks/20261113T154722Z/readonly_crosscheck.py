"""One bounded stdlib-only read of this existing job; never loads a model."""
from pathlib import Path
import datetime as dt,hashlib,json,os,pwd,re,socket,subprocess

JOB='letter-a1-q06-20261113-normal-seed42-r0'
RUN='0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5'
BASE=Path('/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113')
J=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/JOB
T=BASE/'formal_seed42/training'
O=BASE/JOB
utc=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
result={'started_utc':utc(),'job_id':JOB,'expected_RUN_COMMIT':RUN,'host':socket.gethostname(),'uid':os.getuid(),'user':pwd.getpwuid(os.getuid()).pw_name,'artifacts':{},'events':{},'checkpoint_checks':[],'processes':{},'unavailable':[]}
assert (result['host'],result['uid'],result['user'])==('zwlab-01',1005,'bwwu')

def read(path,limit=8*1024**2):
    path=Path(path)
    if not path.is_file():result['unavailable'].append(str(path));return None
    before=path.stat()
    with path.open('rb') as f:
        offset=max(0,before.st_size-limit);f.seek(offset);data=f.read(limit)
    after=path.stat()
    result['artifacts'][str(path)]={'bytes_before':before.st_size,'bytes_after':after.st_size,'mtime_ns_before':before.st_mtime_ns,'mtime_ns_after':after.st_mtime_ns,'read_offset':offset,'observed_bytes':len(data),'observed_bytes_sha256':hashlib.sha256(data).hexdigest(),'capture_atomic':False}
    return data

def obj(path):
    b=read(path)
    return json.loads(b) if b is not None else None

result['persisted_runner_status']=obj(J/'status.json')
result['pipeline']=obj(J/'output/pipeline.json')
req=obj(J/'request.json')
result['request_excerpt']={k:req[k] for k in ['job_id','capture_version','cwd','argv','timeout_seconds','host','user','uid']}
assert req['capture_version']==RUN
result['formal_ledger']=obj(BASE/'formal_seed42_ledger.json')
selected={'train_begin','loader','epoch_begin','epoch_end','validation_complete','checkpoint_complete','strict_ds_restore_begin','strict_ds_restore_complete','strict_ds_restore_end','checkpoint_restored','training_complete','log'}
for rank in [0,1]:
    report=obj(T/f'rank{rank}.json')
    keys=['schema','status','rank','world_size','local_rank','run_commit','config_sha256','argv','interpreter','cwd','model_class','sharing','lora','effective_loader','initial_global_step','optimizer_membership_exact','scheduler_class','deepspeed_owns_scheduler','global_step','epoch','best_checkpoint','best_metric','metrics','elapsed_seconds','error','traceback','peak_allocated_bytes','peak_reserved_bytes']
    result.setdefault('rank_reports',{})[str(rank)]={k:report[k] for k in keys if k in report}
    path=T/f'events.rank{rank}.jsonl';data=read(path);lines=data.splitlines(keepends=True);rows=[];parsed=[]
    if result['artifacts'][str(path)]['read_offset']:lines=lines[1:]
    for line in lines:
        if not line.endswith(b'\n'):continue
        v=json.loads(line);parsed.append((line,v))
    updates=[v for _,v in parsed if v['event']=='optimizer_step']
    # Preserve all phase/validation/checkpoint events and the latest 40 updates/logs.
    last_updates={id(v) for _,v in [(b'',x) for x in updates[-40:]]}
    recent_logs=[v for _,v in parsed if v['event']=='log'][-40:]
    last_logs={id(v) for v in recent_logs}
    for line,v in parsed:
        if (v['event'] in selected and v['event']!='log') or id(v) in last_updates or id(v) in last_logs:rows.append(line.decode())
    result['events'][str(rank)]={'source':str(path),'raw_jsonl_excerpt':''.join(rows),'latest_optimizer_step':updates[-1] if updates else None,'validation':[v for _,v in parsed if v['event']=='validation_complete'],'checkpoint_events':[v for _,v in parsed if v['event']=='checkpoint_complete'],'recent_optimizer_steps':updates[-40:],'recent_train_logs':recent_logs,'parsed_records_in_read_window':len(parsed)}

for cp in sorted(T.glob('checkpoint-*')):
    seal=obj(cp/'complete.json')
    check={'path':str(cp),'seal_present':seal is not None,'status':'INCOMPLETE_NO_SEAL' if seal is None else 'CHECKING','files':[]}
    if seal:
        state=obj(cp/'trainer_state.json');check['seal']=seal;check['trainer_state']=state
        ok=seal['run_commit']==RUN and seal['collective_ranks']==2 and seal['exclude_frozen_parameters'] is False
        expected_config=result['rank_reports']['0']['config_sha256'];ok=ok and seal['config_sha256']==expected_config
        names={x['path'] for x in seal['files']};s=seal['global_step'];prefix=f'global_step{s}/'
        required={'adapter_model.bin','adapter_config.json','trainer_state.json','training_args.bin','external_scheduler.pth','latest','tokenizer.json','tokenizer_config.json','special_tokens_map.json','loader_state_0.pth','loader_state_1.pth','rng_state_0.pth','rng_state_1.pth',prefix+'mp_rank_00_model_states.pt',prefix+'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',prefix+'bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt'}
        ok=ok and required<=names and state['global_step']==s
        for row in seal['files']:
            rel=Path(row['path']);safe=not rel.is_absolute() and '..' not in rel.parts
            p=cp/rel;exists=safe and p.is_file() and not p.is_symlink();size=p.stat().st_size if exists else None
            valid=exists and size==row['bytes'] and size>0
            item={'path':row['path'],'expected_bytes':row['bytes'],'actual_bytes':size,'size_and_file_identity_match':bool(valid),'sha256_checked':False}
            if row['sha256'] is not None and valid:
                # Match the accepted seal contract; never hash large checkpoint payloads.
                assert size<8*1024**2
                actual=hashlib.sha256(p.read_bytes()).hexdigest();item.update(sha256_checked=True,actual_sha256=actual,hash_match=actual==row['sha256']);valid=valid and item['hash_match']
            check['files'].append(item);ok=ok and valid
        latest=(cp/'latest').read_text().strip();ok=ok and latest==prefix.rstrip('/')
        check.update(status='PASS_SEAL_CONTRACT' if ok else 'FAIL_SEAL_CONTRACT',member_count=len(seal['files']),logical_bytes=sum(x['bytes'] for x in seal['files']),large_payload_content_rehashed=False,latest_tag=latest)
    result['checkpoint_checks'].append(check)

result['best_selection']=obj(T/'best_selection.json')
for rank in [0,1]:result.setdefault('test_rank_reports',{})[str(rank)]=obj(O/'test'/f'rank{rank}.json')
result['original_metrics']=obj(O/'test/metrics.json')
result['offline_pipeline']=obj(O/'offline_evaluation/pipeline.json')
for name in ['metrics.json','bootstrap.json','groups.json']:
    p=O/'offline_evaluation/primary'/name
    result.setdefault('extended_existing_results',{})[name]=obj(p)
result['paired_existing_results']=obj(O/'offline_evaluation/paired/comparison.json')

b=read(J/'work.log',2*1024**2);text=b.decode('utf-8','replace');result['worklog_tail']='\n'.join(text.splitlines()[-80:])+'\n'
pattern=re.compile(r'Traceback|CUDA out of memory|OutOfMemoryError|CUMULATIVE_BUDGET_EXHAUSTED|HOST_MEMORY_RESERVE|DATA_DISK_RESERVE|NO_TASK_LOG_PROGRESS|\b(?:nan|inf)\b',re.I)
result['anomaly_lines_in_scanned_worklog']=[x for x in text.splitlines() if pattern.search(x)]
b=read(J/'output/resources.jsonl');resources=[json.loads(x) for x in b.splitlines() if x.strip()]
result['resource_samples_tail']=resources[-5:]
result['resource_window_summary']={'samples':len(resources),'min_mem_available':min(x['mem_available'] for x in resources),'max_swap_used':max(x['swap_used'] for x in resources),'min_swap_used':min(x['swap_used'] for x in resources),'min_disk_available':min(x['disk_available'] for x in resources),'max_seconds_without_worklog_progress':max(x['seconds_without_worklog_progress'] for x in resources)}
status=result['persisted_runner_status'];ids={x['pid'] for x in status.get('worker_group',{}).get('known_members',[])}
ids.update(status[k]['pid'] for k in ['supervisor','worker'] if status.get(k))
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for pid in sorted(ids):
    p=Path('/proc')/str(pid)
    if not p.exists():result['processes'][str(pid)]={'alive':False};continue
    st=(p/'stat').read_text();f=st[st.rfind(')')+2:].split()
    item={'alive':f[0]!='Z','pid':pid,'uid':p.stat().st_uid,'state':f[0],'ppid':int(f[1]),'pgid':int(f[2]),'session_id':int(f[3]),'start_ticks':f[19],'boot_id':boot,'argv':(p/'cmdline').read_bytes().decode().split('\x00')[:-1],'fds':{}}
    for i in [0,1,2]:
        try:item['fds'][str(i)]=os.readlink(p/'fd'/str(i))
        except OSError:pass
    result['processes'][str(pid)]=item
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv'],text=True)
result['compute_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
result['ram']=subprocess.check_output(['free','-b'],text=True)
result['disk']=subprocess.check_output(['df','-B1',str(BASE)],text=True)
result['finished_utc']=utc();result['scope_limits']=['Non-atomic bounded read of an active job','No model/checkpoint tensor loading or GPU computation','Checkpoint seal: all member sizes, required roles, and only pre-sealed small-file hashes; large payload content not rehashed','Existing results only; no generation, evaluation, bootstrap or independent recomputation started']
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
