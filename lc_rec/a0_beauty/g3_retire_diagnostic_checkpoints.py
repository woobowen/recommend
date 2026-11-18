#!/usr/bin/env python3
"""Retire verified, no-longer-consumed diagnostic checkpoint copies only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('old-release','training-root','completed-job-id','comparison-receipt','runner','runner-sha256','output'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.training_root)
    assert not root.is_symlink() and root.name=='recovery'
    assert root.resolve().parent.parent==Path('/data/bwwu/lc-rec/g3-preparation-single-20261118')
    assert root.resolve()==Path('/data/bwwu/lc-rec/g3-preparation-single-20261118/9adaad4ca217/recovery')
    assert a.completed_job_id=='g3-lcrec-single-resume-diagnostic-20261118-r2'
    release=Path(a.old_release);version=json.loads((release/'RUN_VERSION.json').read_text())
    assert version['run_commit']=='9adaad4ca217acd7bdcf5019c3c1eeac6c72db0b'
    assert root.parent.name==version['run_commit'][:12]
    fixed={row['path']:row['sha256'] for row in version['files']}
    for name in ('formal_train.py','g3_identity.py','single_gpu_training.py'):
        source=release/'lc_rec/a0_beauty'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==fixed[str(source)]
    sys.path.insert(0,str(release/'lc_rec/a0_beauty'))
    from formal_train import verify_checkpoint
    from g3_identity import sha,source_identity,verify_adapter_identity
    assert sha(a.runner)==a.runner_sha256
    job=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/a.completed_job_id
    status=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',a.runner,'query',str(job)],text=True))
    assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup_verified']
    assert not status['worker_alive'] and not status['workgroup_live']
    comparison=json.loads(Path(a.comparison_receipt).read_text())
    assert comparison['status']=='PASS' and comparison['scope']=='DIAGNOSTIC_ONLY'
    assert comparison['run_commit']==version['run_commit']
    comparison_job=Path(a.comparison_receipt).parent.parent
    assert comparison_job.parent==job.parent and comparison_job.name=='g3-lcrec-single-resume-compare-diagnostic-20261118-r2'
    comparison_status=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',a.runner,'query',str(comparison_job)],text=True))
    assert comparison_status['status']=='SUCCEEDED' and comparison_status['exit_code']==0 and comparison_status['cleanup_verified']
    assert not comparison_status['worker_alive'] and not comparison_status['workgroup_live']
    assert any(row['path']==str(Path(a.comparison_receipt).relative_to(comparison_job)) and
               row['sha256']==sha(a.comparison_receipt) for row in comparison_status['artifacts'])
    config=release/'lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json'
    assert comparison['config_sha256']==sha(config) and comparison['diagnostic_attempt_job_id']==a.completed_job_id
    preserved=comparison['large_state_identity']+comparison['actual_raw_files']+comparison['row_evidence']
    for member in preserved:
        path=Path(member['path'])
        assert path.is_file() and not path.is_symlink() and path.stat().st_size==member['bytes'] and sha(path)==member['sha256']
    early=root.parent/'early_evidence'
    for name in ('rank0.json','events.rank0.jsonl','failure.rank0.json','epoch1_rows.json','checkpoint-2.complete.json'):
        assert (early/name).is_file() and not (early/name).is_symlink()
    assert json.loads((early/'rank0.json').read_text())['status']=='FAILED'
    provenance=verify_adapter_identity(root/'preflight_adapter',scope='PREFLIGHT',run_commit=version['run_commit'],config_sha256=sha(config))
    assert (root/'boundary_pre_best_rank0.pt').is_file()
    archive=root/'verified_checkpoint_manifests';archive.mkdir(exist_ok=True)
    record={'scope':'VERIFIED_TEMPORARY_DIAGNOSTIC_RETIREMENT','status':'RUNNING',
        'argv':sys.argv,'cwd':str(Path.cwd()),'python':sys.executable,'run_commit':version['run_commit'],
        'source':source_identity([__file__,release/'lc_rec/a0_beauty/formal_train.py']),
        'completed_runner_job':status,'comparison_runner_job':comparison_status,'comparison':source_identity([a.comparison_receipt])[0],
        'adapter_identity_sha256':sha(root/'preflight_adapter/g3_identity.json'),
        'preserved_early':source_identity([early/name for name in ('rank0.json','events.rank0.jsonl','failure.rank0.json','epoch1_rows.json','checkpoint-2.complete.json')]),
        'retained':'All original failed job receipts/logs, failed early evidence, complete manifests, selected adapter and both full comparison states.',
        'free_bytes_before':shutil.disk_usage(root).free,'candidates':[],'removed':[]}
    output=Path(a.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    for cp in sorted(root.glob('checkpoint-*')):
        assert cp.parent==root and cp.is_dir() and not cp.is_symlink()
        meta=verify_checkpoint(cp,version['run_commit'],sha(config))
        assert meta['scope']=='PREFLIGHT' and meta['world_size']==1
        assert cp.name=='checkpoint-'+str(meta['global_step'])
        actual=list(cp.rglob('*'));assert not any(member.is_symlink() for member in actual)
        assert {str(member.relative_to(cp)) for member in actual if member.is_file()}=={row['path'] for row in meta['files']}|{'complete.json'}
        if str(cp)==provenance['checkpoint_source']['path']:
            assert sha(cp/'complete.json')==provenance['checkpoint_source']['complete_sha256']
        manifest=archive/(cp.name+'.complete.json')
        if manifest.exists():assert manifest.read_bytes()==(cp/'complete.json').read_bytes()
        else:shutil.copyfile(cp/'complete.json',manifest)
        allocated=sum(f.stat().st_blocks*512 for f in cp.rglob('*') if f.is_file() and not f.is_symlink())
        record['candidates'].append({'path':str(cp),'manifest':source_identity([manifest])[0],
            'verified_members':len(meta['files']),'allocated_file_blocks_bytes':allocated})
    assert len(record['candidates'])==2 and any(row['path']==provenance['checkpoint_source']['path'] for row in record['candidates'])
    output.write_text(json.dumps(record,indent=2)+'\n')
    for candidate in record['candidates']:
        shutil.rmtree(candidate['path'])
        record['removed'].append(candidate)
        output.write_text(json.dumps(record,indent=2)+'\n')
    record.update(status='PASS',free_bytes_after=shutil.disk_usage(root).free,
        free_space_note='Filesystem free-space delta can include concurrent external writes; removed allocated file blocks are recorded separately.')
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'PASS','removed':len(record['removed']),'output':str(output)}))


if __name__=='__main__':main()
