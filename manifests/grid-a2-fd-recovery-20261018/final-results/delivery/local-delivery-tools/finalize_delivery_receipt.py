#!/usr/bin/env python3
"""Prepare final wbw delivery receipts only after the actual independent ZIP audit."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

BASE=Path(__file__).resolve().parent
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
ROOT=REPO/'manifests/grid-a2-fd-recovery-20261018/final-results'
SCIENTIFIC_COMMIT='42ac9acf23ae0bdc65ed6505089e1b88101755cd'

def load(path):return json.loads(path.read_text())
def ident(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    target=ROOT/'delivery';assert not target.exists()
    request=load(BASE/'deliveries/package-build-request.json');archive=Path(request['output'])
    package=load(archive.with_suffix('.zip.verification.json'))
    windows=load(archive.with_suffix('.zip.windows-access.json'))
    peer_path=BASE/'reviews/delivery_tools/final-package-nonauthor-review.json';peer=load(peer_path)
    assert peer['status']=='FINAL_PACKAGE_AND_SCIENTIFIC_PUBLICATION_NONAUTHOR_REVIEW_PASS'
    assert package['status']=='PACKAGE_INTEGRITY_PASS' and windows['status']=='WINDOWS_UNC_FILE_AND_SHA256_ACCESS_PASS'
    assert package['sha256']==windows['sha256'] and archive.stat().st_size==package['bytes']==windows['bytes']
    assert package['sha256'] in json.dumps(peer), 'Independent report must identify the actual production ZIP.'
    assert package['members']==3153 and package['scientific_review_status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS'
    public=load(BASE/'publication/scientific/anonymous-readback.json');assert public['commit']==SCIENTIFIC_COMMIT
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,check=True,stdout=subprocess.PIPE,text=True).stdout.strip()
    assert head==SCIENTIFIC_COMMIT,'Protect unexpected local changes or external branch integration.'
    scene=load(BASE/'final-scene/scene-20261018T043428336611Z.json')
    target.mkdir(parents=True);records=[]
    def copy(source,relative):
        data=source.read_bytes();path=target/relative;path.parent.mkdir(parents=True,exist_ok=True)
        assert not path.exists();path.write_bytes(data)
        records.append(dict(source=str(source),path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    copy(peer_path,'final-package-nonauthor-review.json')
    for p in sorted(peer_path.parent.glob('final-package-nonauthor*.md')):copy(p,p.name)
    copy(archive.with_suffix('.zip.verification.json'),'package-integrity-verification.json')
    copy(archive.with_suffix('.zip.windows-access.json'),'Windows-UNC-access.json')
    copy(archive.with_suffix('.zip.sha256'),'FINAL_ZIP.sha256')
    member_manifest=BASE/'deliveries/final-evidence-staging-v2/manifest.json'
    assert hashlib.sha256(member_manifest.read_bytes()).hexdigest()==package['manifest_sha256']
    copy(member_manifest,'ZIP_MEMBER_MANIFEST.json')
    for p in sorted((BASE/'publication/scientific').iterdir()):
        if p.is_file():copy(p,'scientific-publication-readback/'+p.name)
    copy(BASE/'publication/precommit-check.json','scientific-publication-readback/precommit-check.json')
    for name in ['build_final_package.py','verify_windows_zip_access.py','verify_source_at_release.py',
                 'append_published_release.py','finalize_delivery_receipt.py']:
        copy(BASE/name,'local-delivery-tools/'+name)
    receipt=dict(status='ALL_SCIENTIFIC_AND_ACTUAL_PACKAGE_REVIEWS_PASS_READY_FOR_FINAL_RECEIPT_PUBLICATION',
      prepared_utc=dt.datetime.now(dt.timezone.utc).isoformat(),GPT_UPLOAD_REQUIRED='YES',
      calculations_and_true_independent_reviews_complete=True,
      remaining_action='Publish this bounded receipt-only commit to wbw and actually read back its ref/key bytes; report that full SHA outside this self-referential file.',
      repository='woobowen/recommend',branch='wbw',scientific_release_commit=SCIENTIFIC_COMMIT,
      scientific_execution_commit=scene['source_commit'],server_snapshot=scene['source_snapshot'],
      server_snapshot_sha256=scene['source_snapshot_sha256'],source_tree_unmodified_by_receipt=True,
      unique_final_ZIP=dict(WSL_absolute_path=str(archive),Windows_UNC_path=windows['Windows_UNC_path'],
                          bytes=package['bytes'],members=package['members'],payload_members=package['payload_members'],
                          sha256=package['sha256'],manifest_sha256=package['manifest_sha256'],
                          member_size_SHA256_CRC_pass=True,path_duplicate_and_case_collision_checks_pass=True,
                          sensitive_patterns_all_member_bytes_pass=True,Windows_actual_binary_read_and_SHA_pass=True,
                          one_actual_production_ZIP=True,ZIP_repacked_after_review=False),
      true_non_author_package_review=ident(peer_path),
      true_reviewers=['/root/fd_repair_reviewer','/root/sports_a1_result_audit','/root/toys_result_audit'],
      scientific_results=load(BASE/'final-results/scientific-results.json')['A_route_original_metrics'],
      completed_A2_new_stages=scene['resources']['stage_details'],activity=scene['activity'],
      total_GPU_requested_seconds=scene['resources']['total_gpu_requested_seconds'],
      approved_GPU_requested_seconds=scene['resources']['total_budget_gpu_seconds'],
      remaining_GPU_requested_seconds=scene['resources']['remaining_gpu_seconds'],
      absolute_deadline_unchanged=scene['resources']['original_absolute_deadline'],
      original_failure_and_old34_history_unchanged=True,new_formal_quantization_attempts=1,
      no_more_experiment_submission_authorized=True,
      resume_readonly_query=['python3','/home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py',
                             '--target','new-server','query','ga-20261018-a2-fd-queue-v1'],
      limits=['Historical Toys producer not unique; do not select/drop a candidate by testing results',
              'A2 method reproduction and new scratch generator, not recovered historical Sports weights',
              'A1/A2 are not a single-factor causal SID comparison',
              'Single seed/fixed32 evaluation cohort and descriptive query statistics do not establish strict invariance or unused target prefix',
              'Full environment per-file mirror/member count/size/hash not captured; byte-identical restore not guaranteed',
              'Original FD worker attribution inferred; FD168 is a sampled maximum, not a continuous upper bound'],
      additions=dict(system_packages=[],language_packages=[],toolchains=[],global_configuration_changes=[],
                     job_scoped_RLIMIT_only=True),
      stop_after_final_readback=True,no_other_branch_sync=True,no_formal_evidence_cleanup=True,
      webpage_GPT_second_review_pending_user_upload=True)
    (target/'DELIVERY.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (target/'copy-manifest.json').write_text(json.dumps(dict(files=records),indent=2)+'\n')
    text='''# 唯一最终ZIP与真实审核回执

全部已授权科研计算与真实非作者审核已通过；实际唯一ZIP也已由主控、真实非作者及Windows UNC路径读数独立核验。详见DELIVERY.json、final-package-nonauthor-review.json与package-integrity-verification.json。

本目录是在唯一ZIP序列化后新增的轻量回执，不能回填进原ZIP制造循环hash。ZIP固定绑定科研发表42ac9acf23ae0bdc65ed6505089e1b88101755cd，实际运行仍85b77f5e2a5b497fdc1f620649ba488fc783a245 / snapshot22286966cd544773。晚回执提交仅增加本目录，不改任何执行源码或科研值。其实际完整提交SHA与远端回读在发布后由最终交付回复/本地回读回执给出，不在此预写自己的commit SHA。

GPT_UPLOAD_REQUIRED: YES。用户只需上传DELIVERY.json中那个唯一ZIP；本目录回执和科研报告可由网页GPT从固定GitHub取得，无需再上传第二包。整包SHA在FINAL_ZIP.sha256及最终回复中；所有大模型/TFRecord留服务器，完整身份与环境缺口在最终ZIP的assets-not-copied.json。无新实验、其他分支同步或清理授权。
'''
    (target/'README.md').write_text(text)
    print(json.dumps(dict(status=receipt['status'],target=str(target),ZIP_sha256=package['sha256'],
                         no_ZIP_or_stage_bytes_modified=True,copied_receipts=len(records)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
