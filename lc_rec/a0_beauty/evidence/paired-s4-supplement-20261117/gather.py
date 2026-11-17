"""Collect small finalized evidence; no statistical computation or original writes."""
import datetime as dt,hashlib,json,resource,tarfile,time
from pathlib import Path
START=time.monotonic();B=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115');S=B/'formal-supplements/s4-paired-seed42';R=B/'formal-candidate-5b2917d/formal-run';J=B/'formal-candidate-5b2917d/jobs/letter-a0-beauty-supp-s4-paired-20261117-a0';load=lambda p:json.loads(Path(p).read_text())
status=load(J/'status.json');assert status.get('ended_at') and status.get('cleanup',{}).get('verified'),'not finalized/cleanup pending'
plan=load(R/'plan.json');files={}
def add(p,name):
 p=Path(p)
 if p.is_file():assert not p.is_symlink();files[name]=p
for name in ('beauty_supplemental_final_summary.json','public_metric_matrices.npz','output-manifest.json','budget.json','original-preservation.json'):
 add(S/name,'results/'+name)
for name in ('paired_comparison.json','paired_bootstrap_difference_means.npy'):add(S/'s4'/name,'results/s4/'+name)
for name in ('request.json','accepted.json','status.json','cleanup.json','attempt.json','work.log','supervisor.log'):add(J/name,'runner/'+name)
for name in ('actual-argv.json','protected-inputs.json','input-manifest.json','prelaunch.json','request.json'):add(S/name,'run/'+name)
for name in ('plan.json','plan.identity.json','state.json','controller.log'):add(R/name,'original/control/'+name)
failed=Path(plan['job_root'])/'letter-a0-beauty-formal-beauty-a0-seed42-formal-paired-a0'
for name in ('request.json','accepted.json','status.json','cleanup.json','work.log'):add(failed/name,'original/failed-paired/'+name)
for name in ('paired_comparison.json','paired_bootstrap_difference_means.npy'):add(R/'attempts/paired/a0/s3'/name,'original/s3/'+name)
for cond in ('normal','s3','s4'):
 for name in ('metrics.json','groups.json','bootstrap.json','bootstrap_draw_means.npy'):add(R/f'attempts/{cond}_eval/a0/primary'/name,f'original/{cond}-primary/'+name)
 for rank in (0,1):add(R/f'attempts/{cond}/a0/test/rank{rank}.json',f'original/{cond}-generation/rank{rank}.json')
for name in ('best_selection.json','rank0.json','rank1.json','events.rank0.jsonl','events.rank1.jsonl'):add(R/'training'/name,'original/training/'+name)
for p in (R/'training/checkpoint_receipts').glob('*.json'):add(p,'original/training/checkpoint_receipts/'+p.name)
for name in ('train','verify_best_adapter','normal','normal_eval','s3','s3_eval','s4','s4_eval'):
 add(R/f'attempts/{name}/a0/stage_result.json',f'original/stage-results/{name}.json')
for name in ('formal_control.py','production_control.py','production_pipeline.py','production_identity.py','paired_beauty.py','offline_metrics.py','formal_train.py','checkpoint_receipt.py'):
 add(Path(plan['release'])/'lc_rec/a0_beauty'/name,'frozen-source/lc_rec/a0_beauty/'+name)
for rel in ('lc_rec/letter/review/final_audit_20261113/statistics_verify.py','lc_rec/letter/review/final_audit_20261113/external/official_cce.py','lc_rec/a0_beauty/config/formal_beauty_seed42.json'):
 add(Path(plan['release'])/rel,'frozen-source/'+rel)
manifest=[{'member':n,'source':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(files.items())]
archive=B/'formal-supplements/s4-paired-evidence-20261117.tar'
with tarfile.open(archive,'x') as t:
 for n,p in sorted(files.items()):t.add(p,arcname=n,recursive=False)
with (S/'collection-manifest.json').open('x') as f:json.dump({'files':manifest,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-START,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime},f,indent=2)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':len(files),'status':status['status']}))
