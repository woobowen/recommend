from pathlib import Path
import ast,collections,hashlib,json,subprocess
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');base='e8fc4289b4e6c0d8ad00aaa8890f5e50b04b31a4'
commit='e9a3a631929600757b7a99b5763540c6dd9d2b99';e=repo/'lc_rec/a0_beauty/evidence/production-closure';r=e/'final-server';run=r/'SYNTHETIC_CPU_INTEGRATION_ONLY_e9a3a6319296'
def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
def load(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert git('cat-file','-t',commit).decode().strip()=='commit'
v=load(r/f'releases/{commit}/RUN_VERSION.json')
for row in v['files']:
 b=git('show',commit+':'+row['relative']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row
state=load(run/'state.json');assert state['status']=='COMPLETE' and len(state['stages'])==10
assert all(s['status']=='COMPLETE' for s in state['stages'].values())
counts=collections.Counter(a['stage'] for a in state['attempts']);assert all(counts[s]==1 for s in ['train','normal','s3','s4'])
a=load(run/'acceptance.json');assert a['status']=='PASS' and not a['gpu_used']
for stage in state['stages']:
 attempts=[x for x in state['attempts'] if x['stage']==stage]
 assert all(x['runner']['exit_code']==0 and x['runner']['cleanup_verified'] for x in attempts)
for c in ('normal','s3','s4'):
 m=load(run/f'attempts/{c}_eval/a0/primary/metrics.json')
 assert m['status']=='PASS' and m['users']==22363 and len(m['metrics'])==16 and m['official_cce_full_user_comparisons']==22363
paired={}
for c in ('s3','s4'):
 d=load(run/f'attempts/paired/a0/{c}/paired_comparison.json');assert d['status']=='PASS' and d['users']==22363
 assert d['bootstrap']['repetitions']==5000 and d['bootstrap']['seed']==20261113
 assert d['independent_verification']['maximum_draw_difference']<1e-13
 paired[c]=d['independent_verification']
conformance=load(run/'concurrent_clients/conformance.json');assert conformance['status']=='PASS'
assert conformance['formal_prepare_and_rejection']['prepare_exit']==0 and conformance['formal_prepare_and_rejection']['submit_without_authorization_exit']==1
unchanged=[]
for p in ['lc_rec/a0_beauty/'+n for n in ('formal_train.py','formal_evaluate.py','offline_metrics.py','shuffle_entry.py','shuffle_runtime.py')]+['lc_rec/shuffle/core.py']:
 b=git('show',commit+':'+p);assert b==git('show',base+':'+p)
 unchanged.append(dict(path=p,sha256=hashlib.sha256(b).hexdigest(),unchanged_from=base))
old=git('show',base+':lc_rec/a0_beauty/paired_beauty.py').decode();new=git('show',commit+':lc_rec/a0_beauty/paired_beauty.py').decode()
def func(s,n):return ast.dump(next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
assert func(old,'paired_bootstrap')==func(new,'paired_bootstrap')
def body(s):return s[s.index('    rows=[];matrices=[]'):s.index('    result={')]
assert body(old)==body(new)
save(e/'method_invariance.json',dict(engineering_commit=commit,unchanged_source_files=unchanged,paired_bootstrap_ast_equal=True,full_paired_calculation_body_byte_equal=True,scope='No numeric-equivalence claim beyond actual tests. Different Beauty Embedding/Head vocabulary rows and previously approved accumulation/recovery remain inherited.',new_method_change='NONE'))
budget=load(r/'budget.json');budget['local_cpu_test_allowance_seconds']=60;budget['accounted_upper_bound_seconds']=budget['wall_seconds_upper_bound_including_nested_jobs']+60
save(e/'INTERNAL_REVIEW.json',dict(status='PASS',author='main controller; NOT non-author review',engineering_commit=commit,server_export_files_verified_against_git=v['files'],pipeline_all_ten_stages_complete=True,stage_accepted_counts=dict(counts),all_three_conditions_users=22363,candidates_each=447260,full_paired_and_independent_draws_verified=paired,fault_suite_cases=len(a['negative_tests']),formal_cli_prepare_exit=0,formal_cli_no_authorization_exit=1,GPU_USED=0,budget=budget,nonauthor_review='PENDING',formal_launch_authorized=False))
(e/'engineering.diff').write_bytes(git('diff',base,commit,'--','lc_rec/a0_beauty/*.py'))
print(json.dumps(dict(status='PASS',engineering_commit=commit,export_files=len(v['files']),stage_counts=dict(counts),fault_cases=len(a['negative_tests']),budget_upper_bound=budget['accounted_upper_bound_seconds'])))
