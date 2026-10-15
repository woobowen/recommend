"""Publish small immutable receipts into the existing stage manifest only."""
import json
from pathlib import Path
import shutil

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
ROOT=REPO/'manifests/grid-formal-20261015'


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.read_bytes()==source.read_bytes(),str(target)
    else:shutil.copyfile(source,target)


def main():
    summary_path=ROOT/'EXECUTION_SUMMARY.json'
    summary=json.loads(summary_path.read_text())
    records={r['job_id']:r for r in summary['jobs']}
    metrics={}
    for directory in sorted(JOBS.glob('gf-20261015-*')):
        retrieved=directory/'retrieved'
        if not (retrieved/'status.json').is_file():continue
        request=json.loads((retrieved/'request.json').read_text())
        assert request['cwd'].startswith('/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015/snapshots/')
        status=json.loads((retrieved/'status.json').read_text())
        assert status['status'] in ('SUCCEEDED','FAILED','TIMED_OUT') and status['cleanup']['verified']
        for source in retrieved.rglob('*'):
            if source.is_file() and not source.is_symlink() and source.suffix not in ('.jsonl','.ckpt','.pt','.pth','.pkl') and source.stat().st_size<=1024**2:
                copy(source,ROOT/'runs'/directory.name/source.relative_to(retrieved))
        phase=request['argv'][4] if 'experiments.formal_20261015.run' in request['argv'] else 'CPU_CONFIGURATION_REGRESSION'
        record={'job_id':directory.name,'phase':phase,'status':status['status'],'exit_code':status['exit_code'],
            'fetched':True,'source_commit':request['capture_version'].split('/')[0],
            'snapshot_sha256':request['capture_version'].split('/')[1],
            'physical_gpu':request['env']['CUDA_VISIBLE_DEVICES'],'cleanup_verified':True,
            'started_at':status['started_at'],'ended_at':status['ended_at'],
            'evidence':'runs/'+directory.name,
            'full_export_local':str(retrieved),'full_export_remote':'/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+directory.name}
        records[directory.name]=record
        result=retrieved/'output/RESULT.json'
        if result.exists():
            value=json.loads(result.read_text())
            if value.get('formal') and value.get('test_complete') and 'route' in value and 'metrics' in value:
                metrics[value['route']+'-'+str(value['shuffle_n'])]={'job_id':directory.name,**value['metrics']}
    for source in (SUPPORT/'results-review-r4').glob('*formal-review.json'):
        copy(source,ROOT/'results-review'/source.name)
    for name in ('collector-review.json',):
        source=SUPPORT/name
        if source.exists():copy(source,ROOT/name)
    summary['jobs']=list(records.values());summary['beauty_formal_metrics']=metrics
    summary['formal_long_jobs_started']=bool(metrics)
    summary['status']='BEAUTY_FORMAL_EVALUATIONS_COMPLETE_DOMAIN_METADATA_PENDING' if len(metrics)==6 else 'BEAUTY_FORMAL_EVALUATING'
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'completed_fetched_jobs':len(records),'completed_Beauty_formal_conditions':len(metrics)}))


if __name__=='__main__':main()
